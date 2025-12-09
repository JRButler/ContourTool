#!/usr/bin/env python3
"""
Geodesic contour extraction using fast marching method.
Computes distance maps from source points and extracts isolines.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
from skimage import io, color
from skimage.filters import gaussian
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq

# Optional scikit-fmm support (C implementation, much faster)
try:
    import skfmm
    HAS_SKFMM = True
except ImportError:
    HAS_SKFMM = False

# Optional progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate geodesic contours from an image using fast marching.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jpg -o output.svg
  %(prog)s input.jpg --source 100,200 --source 300,400 --num 50
  %(prog)s input.jpg --source 100,200,300,400,500,600 --num 50
  %(prog)s input.jpg --gamma 1.5 --blur 2.0 --thickness 2
  %(prog)s input.jpg --dark-boost 2.0 --bright-cut 0.7 --num 120
        """
    )
    
    parser.add_argument('input', type=str,
                        help='Input image path')
    parser.add_argument('-o', '--output', type=str, default='output.svg',
                        help='Output image path (default: output.svg)')
    parser.add_argument('--source', type=str, action='append',
                        help='Source point(s) as "x1,y1" or "x1,y1,x2,y2,..." for multiple points. '
                             'Can be specified multiple times. If not specified, uses image center.')
    parser.add_argument('--num', type=int, default=30,
                        help='Number of contour levels (default: 30)')
    parser.add_argument('--min', type=int, default=10,
                        help='Minimum number of points per contour (default: 10)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma correction factor (default: 1.0)')
    parser.add_argument('--dark-boost', type=float, default=1.0,
                        help='Multiply intensity in dark regions (<40%% brightness)\n'
                             '  • >1.0 → MORE LINES in shadows (great for faces, hair)\n'
                             '  • <1.0 → suppresses dark noise\n'
                             '  • Try: 1.5–2.5 for portraits, 1.0 for flat art')
    parser.add_argument('--bright-cut', type=float, default=1.0,
                        help='Cap intensity in bright regions (>70%% brightness)\n'
                             '  • <1.0 → REDUCES LINES in highlights (clean sky, white areas)\n'
                             '  • =1.0 → no change\n'
                             '  • Try: 0.6–0.8 to avoid over-plotting in bright zones')
    parser.add_argument('--blur', type=float, default=0.0,
                        help='Gaussian blur sigma for input image (default: 0.0)')
    parser.add_argument('--smooth', type=float, default=2.0,
                        help='Smooth distance field to reduce grid artifacts (default: 2.0)\n'
                             '  • 0 = no smoothing (may show diagonal artifacts)\n'
                             '  • 1-3 = recommended range for clean contours')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for input image (default: 1.0)')
    parser.add_argument('--thickness', type=float, default=1.0,
                        help='Line thickness for contours (default: 1.0)')
    parser.add_argument('--color', type=str, default='black',
                        help='Contour color (default: black)')
    parser.add_argument('--bg', type=str, default='white',
                        help='Background color (default: white)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')
    parser.add_argument('--clean-artifacts', action='store_true', default=True,
                        help='Split paths at MOVETO codes to remove jump artifacts (default: on)')
    parser.add_argument('--no-clean-artifacts', action='store_false', dest='clean_artifacts',
                        help='Disable path splitting')
    parser.add_argument('--pure-python', action='store_true',
                        help='Use pure Python fast marching (slower, for comparison/fallback)')
    parser.add_argument('--progress', action='store_true',
                        help='Show progress bar (requires tqdm)')
    
    return parser.parse_args()


def load_and_preprocess(image_path, args):
    """Load image and apply preprocessing steps - matches original behavior."""
    # Load image
    img = io.imread(image_path, as_gray=True).astype(np.float32)
    
    # Normalize to 0-1 range
    if img.max() > 1.0:
        img /= 255.0
    
    # Scale if requested
    if args.scale != 1.0:
        new_shape = (int(img.shape[0] * args.scale), int(img.shape[1] * args.scale))
        img = resize(img, new_shape, anti_aliasing=True, preserve_range=True)
    
    # Apply Gaussian blur FIRST (like original)
    if args.blur > 0:
        img = gaussian(img, sigma=args.blur, preserve_range=True)
    
    # Store as intensity for further processing
    intensity = img.copy()
    
    # Apply gamma correction
    if args.gamma != 1.0:
        intensity = np.power(intensity, args.gamma)
        # Re-normalize after gamma (critical for preserving curve)
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    
    # Apply dark boost and bright cut
    if args.dark_boost != 1.0 or args.bright_cut != 1.0:
        mask_dark = intensity < 0.4
        mask_bright = intensity > 0.7
        intensity[mask_dark] *= args.dark_boost
        intensity[mask_bright] = np.clip(intensity[mask_bright], 0.0, args.bright_cut)
    
    return intensity


def build_speed_map(intensity):
    """Convert intensity to speed map - matches original slowness calculation."""
    # Original: slowness = max(1.0 - intensity, epsilon), then speed = 1/slowness
    # This creates strong non-linear response preserving midtones
    # Clamp minimum slowness to 0.01 to prevent extreme speed ratios (max 100:1)
    # This eliminates numerical artifacts in very bright regions
    slowness = np.maximum(1.0 - intensity, 0.01)
    speed = 1.0 / slowness
    return speed


def fast_marching_skfmm(speed, sources):
    """
    Fast Marching using scikit-fmm (C implementation).
    
    Much faster than pure Python - typically 10-50x speedup.
    
    Args:
        speed: 2D array of propagation speeds
        sources: List of (x, y) tuples for starting points
    
    Returns:
        T: Distance/travel-time map from sources
    """
    h, w = speed.shape
    
    # Create phi array: negative at sources, positive elsewhere
    # skfmm treats the zero crossing as the boundary
    phi = np.ones((h, w), dtype=np.float64)
    
    for x, y in sources:
        if 0 <= x < w and 0 <= y < h:
            phi[y, x] = -1.0
    
    # Compute travel time from sources
    # skfmm.travel_time solves |grad(T)| * speed = 1
    T = skfmm.travel_time(phi, speed)
    
    return T


def fast_marching_dijkstra(speed, sources):
    """
    Fast Marching Method using proper upwind scheme for Eikonal equation.
    Produces smooth circular geodesic contours.
    
    Args:
        speed: 2D array of propagation speeds
        sources: List of (x, y) tuples for starting points
    
    Returns:
        T: Distance map from sources
    """
    h, w = speed.shape
    T = np.full((h, w), np.inf, dtype=np.float64)
    
    # Status: 0=far, 1=narrow band, 2=frozen
    status = np.zeros((h, w), dtype=np.uint8)
    
    # Initialize all source points
    # Use 4-connected neighbors only to match Eikonal solver (prevents grid artifacts)
    neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    heap = []
    for x, y in sources:
        if 0 <= x < w and 0 <= y < h:
            T[y, x] = 0.0
            status[y, x] = 2  # frozen
            # Add neighbors to narrow band
            for dy, dx in neighbors_4:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and status[ny, nx] == 0:
                    status[ny, nx] = 1
                    T[ny, nx] = solve_eikonal(T, speed, ny, nx, h, w)
                    heapq.heappush(heap, (T[ny, nx], ny, nx))
    
    if not heap:
        raise ValueError("No valid source points provided")
    
    processed = 0
    total_pixels = h * w
    
    # Progress tracking
    show_progress = False
    pbar = None
    if HAS_TQDM and '--progress' in sys.argv:
        show_progress = True
        pbar = tqdm(total=total_pixels, desc="Fast marching", unit="px")
    
    while heap:
        dist, y, x = heapq.heappop(heap)
        
        # Skip if already frozen or if we have a better value
        if status[y, x] == 2 or dist > T[y, x]:
            continue
        
        # Freeze this point
        status[y, x] = 2
        processed += 1
        
        if show_progress and pbar:
            pbar.update(1)
        elif not show_progress and processed % 10000 == 0:
            print(f"\rProgress: {100 * processed / total_pixels:.1f}%", end='', flush=True)
        
        # Update 4-connected neighbors only (matches Eikonal solver)
        for dy, dx in neighbors_4:
            ny, nx = y + dy, x + dx
            
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            
            if status[ny, nx] == 2:  # already frozen
                continue
            
            # Solve Eikonal equation for this neighbor
            T_new = solve_eikonal(T, speed, ny, nx, h, w)
            
            if T_new < T[ny, nx]:
                T[ny, nx] = T_new
                if status[ny, nx] == 0:  # far -> narrow band
                    status[ny, nx] = 1
                    heapq.heappush(heap, (T_new, ny, nx))
                else:  # already in narrow band, update
                    heapq.heappush(heap, (T_new, ny, nx))
    
    if show_progress and pbar:
        pbar.close()
    elif not show_progress:
        print("\rProgress: 100.0%")
    
    return T


def solve_eikonal(T, speed, y, x, h, w):
    """
    Solve the Eikonal equation |∇T| = 1/F using upwind finite differences.
    This produces smooth circular geodesics.
    
    The Eikonal equation: |∇T|² = 1/F²
    Discretized: max(Dx-, 0)² + min(Dx+, 0)² + max(Dy-, 0)² + min(Dy+, 0)² = 1/F²
    """
    F = speed[y, x]
    
    # Get upwind differences in x and y directions
    # Use minimum values from frozen neighbors (upwind scheme)
    
    # X direction
    T_xmin = np.inf
    if x > 0 and T[y, x-1] < np.inf:
        T_xmin = min(T_xmin, T[y, x-1])
    if x < w-1 and T[y, x+1] < np.inf:
        T_xmin = min(T_xmin, T[y, x+1])
    
    # Y direction  
    T_ymin = np.inf
    if y > 0 and T[y-1, x] < np.inf:
        T_ymin = min(T_ymin, T[y-1, x])
    if y < h-1 and T[y+1, x] < np.inf:
        T_ymin = min(T_ymin, T[y+1, x])
    
    # Solve quadratic equation for T
    # Case 1: Use both dimensions
    if T_xmin < np.inf and T_ymin < np.inf:
        # (T - T_xmin)² + (T - T_ymin)² = (1/F)²
        # 2T² - 2T(T_xmin + T_ymin) + T_xmin² + T_ymin² - 1/F² = 0
        a = 2.0
        b = -2.0 * (T_xmin + T_ymin)
        c = T_xmin*T_xmin + T_ymin*T_ymin - 1.0/(F*F)
        discriminant = b*b - 4*a*c
        
        if discriminant >= 0:
            T_new = (-b + np.sqrt(discriminant)) / (2*a)
            # Check if solution is valid (upwind condition)
            if T_new >= T_xmin and T_new >= T_ymin:
                return T_new
    
    # Case 2: Use only X dimension
    if T_xmin < np.inf:
        T_new = T_xmin + 1.0/F
        if T_ymin == np.inf or T_new <= T_ymin:
            return T_new
    
    # Case 3: Use only Y dimension
    if T_ymin < np.inf:
        T_new = T_ymin + 1.0/F
        return T_new
    
    # Fallback (should not reach here if properly initialized)
    return np.inf


def split_path_at_moves(path):
    """
    Split a matplotlib Path at MOVETO commands.
    
    Matplotlib contour paths can contain multiple disconnected segments,
    indicated by MOVETO codes. If we just use vertices directly, we get
    straight lines connecting the segments (the "pen down" artifacts).
    
    This function properly splits the path into separate continuous segments.
    
    Args:
        path: matplotlib Path object
    
    Returns:
        List of Path objects, one per continuous segment
    """
    from matplotlib.path import Path as MplPath
    
    vertices = path.vertices
    codes = path.codes
    
    # If no codes, path is a simple continuous polyline
    if codes is None:
        return [path]
    
    # Split at MOVETO codes (code == 1)
    # First point is always MOVETO, so we look for subsequent MOVETOs
    segments = []
    current_start = 0
    
    for i in range(1, len(codes)):
        if codes[i] == MplPath.MOVETO:
            # End current segment, start new one
            if i - current_start >= 2:
                segment_verts = vertices[current_start:i]
                segments.append(MplPath(segment_verts))
            current_start = i
    
    # Don't forget the last segment
    if len(vertices) - current_start >= 2:
        segment_verts = vertices[current_start:]
        segments.append(MplPath(segment_verts))
    
    return segments if segments else [path]


def render_contours(T, args):
    """Render contours directly from distance map (matches original behavior)."""
    h, w = T.shape
    
    # Smooth the distance field to eliminate grid anisotropy artifacts
    # This removes the diamond/crosshatch patterns from fast marching
    if args.smooth > 0:
        from scipy.ndimage import gaussian_filter
        # Only smooth finite values, preserve structure
        T_smooth = T.copy()
        finite_mask = np.isfinite(T) & (T < 1e9)
        # Replace inf temporarily for filtering
        max_finite = T[finite_mask].max() if finite_mask.any() else 0
        T_smooth[~finite_mask] = max_finite
        T_smooth = gaussian_filter(T_smooth, sigma=args.smooth)
        # Restore inf where it was
        T_smooth[~finite_mask] = np.inf
        T = T_smooth
        print(f"Applied distance field smoothing (sigma={args.smooth})")
    
    # Get finite values for percentile calculation
    finite_mask = np.isfinite(T) & (T < 1e9)
    finite_values = T[finite_mask]
    
    if len(finite_values) == 0:
        raise ValueError("No valid travel times to generate contours")
    
    # Safe percentile calculation
    min_T = finite_values.min()
    # Prevent percentile from going out of bounds on small arrays
    percentile_val = min(99.99, 100 * (len(finite_values) - 1) / len(finite_values))
    max_T = np.percentile(finite_values, percentile_val)
    
    print(f"Generating {args.num} contour levels: {min_T:.3f} → {max_T:.3f}")
    
    # Generate contour levels
    levels = np.linspace(min_T, max_T, args.num)
    
    # Draw contours exactly like the original
    print("Drawing and filtering contours...")
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    cs = ax.contour(T, levels=levels, colors=args.color, linewidths=args.thickness)
    
    # Filter contours in place (compatible with matplotlib 3.8+)
    filtered = 0
    artifacts_removed = 0
    
    if hasattr(cs, 'collections'):
        # Old matplotlib (<3.8)
        total_colls = len(cs.collections)
        for idx, coll in enumerate(cs.collections):
            paths = coll.get_paths()
            processed_paths = []
            for p in paths:
                if len(p.vertices) >= args.min:
                    if args.clean_artifacts:
                        split_paths = split_path_at_moves(p)
                        artifacts_removed += len(split_paths) - 1
                        processed_paths.extend([sp for sp in split_paths if len(sp.vertices) >= args.min])
                    else:
                        processed_paths.append(p)
                else:
                    filtered += 1
            coll.set_paths(processed_paths)
            pct = 100 * (idx + 1) / total_colls
            print(f"\rProcessing contours: {pct:.0f}%", end='', flush=True)
        print()  # newline after progress
    else:
        # New matplotlib (3.8+)
        all_paths = cs.get_paths()
        total_paths = len(all_paths)
        processed_paths = []
        for idx, p in enumerate(all_paths):
            if len(p.vertices) >= args.min:
                if args.clean_artifacts:
                    split_paths = split_path_at_moves(p)
                    artifacts_removed += len(split_paths) - 1
                    processed_paths.extend([sp for sp in split_paths if len(sp.vertices) >= args.min])
                else:
                    processed_paths.append(p)
            else:
                filtered += 1
            
            # Progress output every 5% or so
            if total_paths > 0 and (idx + 1) % max(1, total_paths // 20) == 0:
                pct = 100 * (idx + 1) / total_paths
                print(f"\rProcessing contours: {pct:.0f}%", end='', flush=True)
        print(f"\rProcessing contours: 100%")
        
        # Clear existing contours and redraw only the processed paths
        for artist in list(ax.collections):
            artist.remove()
        from matplotlib.collections import LineCollection
        from matplotlib.path import Path as MplPath
        segments = []
        for path in processed_paths:
            segments.append(path.vertices)
        if segments:
            lc = LineCollection(segments, colors=args.color, linewidths=args.thickness)
            ax.add_collection(lc)
    
    print(f"Filtered out {filtered} tiny contours" + 
          (f", split {artifacts_removed} paths at MOVETO jumps" if artifacts_removed > 0 else ""))
    
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with format detection from file extension
    output_format = Path(args.output).suffix[1:].lower() or 'svg'
    print(f"Saving {output_format.upper()}: {args.output}")
    plt.savefig(args.output, format=output_format, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading image: {args.input}")
    img = load_and_preprocess(args.input, args)
    h, w = img.shape
    
    print(f"Image size: {w}x{h}")
    
    # Parse source points or use center
    sources = []
    if args.source:
        for src_str in args.source:
            try:
                # Split by comma and parse all coordinates
                coords = list(map(int, src_str.split(',')))
                
                # Check if we have pairs of coordinates
                if len(coords) % 2 != 0:
                    print(f"Warning: Invalid source format '{src_str}', need pairs of coordinates", 
                          file=sys.stderr)
                    continue
                
                # Add all coordinate pairs as source points
                for i in range(0, len(coords), 2):
                    x, y = coords[i], coords[i + 1]
                    sources.append((x, y))
                    print(f"Source point: ({x}, {y})")
                    
            except ValueError:
                print(f"Warning: Invalid source format '{src_str}', expected numbers", 
                      file=sys.stderr)
    
    # If no valid sources parsed, use center as default
    if not sources:
        center_x, center_y = w // 2, h // 2
        sources = [(center_x, center_y)]
        print(f"Using default source at center: ({center_x}, {center_y})")
    
    # Build speed map
    print("Building speed map...")
    speed = build_speed_map(img)
    
    # Run fast marching - choose implementation
    use_skfmm = HAS_SKFMM and not args.pure_python
    
    if use_skfmm:
        print("Running fast marching (scikit-fmm C implementation)...")
    else:
        if args.pure_python:
            print("Running fast marching (pure Python, --pure-python flag)...")
        else:
            print("Running fast marching (pure Python fallback, install scikit-fmm for 10-50x speedup)...")
    
    start_time = time.time()
    
    if use_skfmm:
        T = fast_marching_skfmm(speed, sources)
    else:
        T = fast_marching_dijkstra(speed, sources)
    
    elapsed = time.time() - start_time
    print(f"Fast marching completed in {elapsed:.2f}s")
    
    # Render output directly from distance map
    render_contours(T, args)
    print("Done!")


if __name__ == '__main__':
    main()