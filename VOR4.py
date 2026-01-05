#!/usr/bin/env python3
"""
VOR4 - Cellular Voronoi Generator
Creates organic cell-like Voronoi patterns with uniform cell sizes,
adjustable wall thickness, and a proper outside border.
"""

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import random

class CellularVoronoi:
    def __init__(self, width=800, height=600, num_cells=100, 
                 wall_thickness_ratio=0.15, seed=None):
        """
        Initialize the Cellular Voronoi generator.
        
        Args:
            width: Canvas width
            height: Canvas height
            num_cells: Approximate number of cells to generate
            wall_thickness_ratio: Wall thickness as ratio of average cell spacing
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.num_cells = num_cells
        self.wall_thickness_ratio = wall_thickness_ratio
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Calculate average spacing for uniform distribution
        area_per_cell = (width * height) / num_cells
        self.avg_spacing = np.sqrt(area_per_cell)
        self.wall_thickness = self.avg_spacing * wall_thickness_ratio
        
    def generate_uniform_random_points(self):
        """
        Generate points with controlled randomness for uniform cell sizes.
        Points fill an organic boundary shape.
        """
        points = []
        
        # Create a denser initial grid to ensure edge coverage
        extra_density = 1.3  # Generate 30% more points initially
        adjusted_cells = int(self.num_cells * extra_density)
        cols = int(np.sqrt(adjusted_cells * self.width / self.height))
        rows = int(adjusted_cells / cols)
        
        x_spacing = self.width / cols
        y_spacing = self.height / rows
        
        # Create organic boundary shape
        center_x = self.width / 2
        center_y = self.height / 2
        
        # Generate grid points
        for i in range(rows):
            for j in range(cols):
                # Grid position with some offset from edges
                x = (j + 0.5) * x_spacing
                y = (i + 0.5) * y_spacing
                
                # Check if point is within organic boundary
                dx = x - center_x
                dy = y - center_y
                angle = np.arctan2(dy, dx)
                distance = np.sqrt(dx**2 + dy**2)
                
                # Create organic boundary using multiple sinusoidal variations
                base_radius_x = self.width * 0.42
                base_radius_y = self.height * 0.42
                base_radius = (base_radius_x + base_radius_y) / 2
                
                # More complex organic variations
                variation1 = np.sin(angle * 3 + 0.5) * 0.12
                variation2 = np.sin(angle * 5 - 0.3) * 0.08
                variation3 = np.cos(angle * 4 + 1.0) * 0.06
                variation4 = np.sin(angle * 7) * 0.04
                
                # Calculate boundary radius at this angle
                boundary_radius = base_radius * (1 + variation1 + variation2 + variation3 + variation4)
                
                # Include points within and slightly outside boundary for edge coverage
                if distance < boundary_radius * 1.15:
                    # Add controlled randomness
                    x += (random.random() - 0.5) * x_spacing * 0.4
                    y += (random.random() - 0.5) * y_spacing * 0.4
                    points.append([x, y])
        
        # Filter to approximately target number
        if len(points) > self.num_cells:
            # Randomly sample to get target number
            indices = random.sample(range(len(points)), self.num_cells)
            points = [points[i] for i in indices]
            
        return np.array(points)
    
    def add_boundary_points(self, points):
        """
        Add mirrored boundary points for proper Voronoi edges.
        """
        extended_points = list(points)
        
        # Add points mirrored across the boundary
        center_x = self.width / 2
        center_y = self.height / 2
        
        for point in points:
            x, y = point
            dx = x - center_x
            dy = y - center_y
            dist = np.sqrt(dx**2 + dy**2)
            
            # Mirror points that are near the boundary
            if dist > min(self.width, self.height) * 0.3:
                # Calculate the mirror point
                scale = 1.4  # Mirror further out
                mirror_x = center_x + dx * scale
                mirror_y = center_y + dy * scale
                extended_points.append([mirror_x, mirror_y])
        
        return np.array(extended_points)
    
    def create_cell_polygons(self, vor, points):
        """
        Create polygon objects for each Voronoi cell.
        """
        cells = []
        
        # Create boundary polygon for clipping
        center_x = self.width / 2
        center_y = self.height / 2
        boundary_points = []
        
        # Generate boundary polygon
        for angle in np.linspace(0, 2 * np.pi, 100):
            base_radius = (self.width * 0.42 + self.height * 0.42) / 2
            
            variation1 = np.sin(angle * 3 + 0.5) * 0.12
            variation2 = np.sin(angle * 5 - 0.3) * 0.08
            variation3 = np.cos(angle * 4 + 1.0) * 0.06
            variation4 = np.sin(angle * 7) * 0.04
            
            radius = base_radius * (1 + variation1 + variation2 + variation3 + variation4)
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            boundary_points.append([x, y])
        
        boundary_polygon = Polygon(boundary_points)
        
        # Process only original points
        for i in range(len(points)):
            if vor.point_region[i] != -1:
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    vertices = [vor.vertices[v] for v in region]
                    poly = Polygon(vertices)
                    
                    # Clip to boundary
                    clipped = poly.intersection(boundary_polygon)
                    
                    if not clipped.is_empty:
                        if isinstance(clipped, Polygon):
                            cells.append(clipped)
                        elif isinstance(clipped, MultiPolygon):
                            # Take the largest polygon
                            largest = max(clipped.geoms, key=lambda p: p.area)
                            cells.append(largest)
        
        return cells
    
    def add_organic_variation(self, cells):
        """
        Add slight organic variations to cell shapes.
        """
        varied_cells = []
        
        for cell in cells:
            if hasattr(cell, 'exterior'):
                coords = []
                exterior_coords = list(cell.exterior.coords)
                
                for i, (x, y) in enumerate(exterior_coords):
                    # Small sinusoidal variations
                    variation = self.avg_spacing * 0.02
                    # Use index to create consistent but varied perturbations
                    dx = variation * np.sin(i * 0.5 + y * 0.01) * (0.5 + random.random() * 0.5)
                    dy = variation * np.cos(i * 0.5 + x * 0.01) * (0.5 + random.random() * 0.5)
                    coords.append((x + dx, y + dy))
                
                try:
                    varied_cell = Polygon(coords)
                    if varied_cell.is_valid:
                        varied_cells.append(varied_cell)
                    else:
                        varied_cells.append(cell)
                except:
                    varied_cells.append(cell)
            else:
                varied_cells.append(cell)
        
        return varied_cells
    
    def generate(self):
        """
        Generate the complete cellular Voronoi pattern.
        """
        # Generate uniform random points
        points = self.generate_uniform_random_points()
        
        # Add boundary points for proper edge handling
        extended_points = self.add_boundary_points(points)
        
        # Create Voronoi diagram
        vor = Voronoi(extended_points)
        
        # Create cell polygons
        cells = self.create_cell_polygons(vor, points)
        
        # Add organic variations
        cells = self.add_organic_variation(cells)
        
        return cells, vor
    
    def visualize(self, cells):
        """
        Visualize the cellular pattern.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), facecolor='white')
        
        # Set white background
        ax.set_facecolor('white')
        
        # Create union of all cells for the overall shape
        all_cells = unary_union(cells)
        
        # Draw black fill for the overall shape (this will be the walls)
        if hasattr(all_cells, 'exterior'):
            black_fill = patches.Polygon(
                list(all_cells.exterior.coords),
                facecolor='black',
                edgecolor='none',
                zorder=0
            )
            ax.add_patch(black_fill)
        
        # Draw individual cells (shrunken to create wall gaps)
        for cell in cells:
            if hasattr(cell, 'exterior'):
                # Shrink cell to create wall thickness
                try:
                    shrunken_cell = cell.buffer(-self.wall_thickness / 2)
                    
                    if not shrunken_cell.is_empty:
                        # Handle both Polygon and MultiPolygon results
                        if isinstance(shrunken_cell, Polygon):
                            polys = [shrunken_cell]
                        elif isinstance(shrunken_cell, MultiPolygon):
                            polys = list(shrunken_cell.geoms)
                        else:
                            continue
                        
                        for poly in polys:
                            if hasattr(poly, 'exterior'):
                                # Slight color variation for organic look
                                gray_value = 0.82 + random.random() * 0.08
                                color = (gray_value, gray_value, gray_value)
                                
                                poly_patch = patches.Polygon(
                                    list(poly.exterior.coords),
                                    facecolor=color,
                                    edgecolor='none',
                                    alpha=1.0,
                                    zorder=1
                                )
                                ax.add_patch(poly_patch)
                except:
                    # If buffering fails, skip this cell
                    pass
        
        ax.set_xlim(-50, self.width + 50)
        ax.set_ylim(-50, self.height + 50)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title('Cellular Voronoi Pattern', fontsize=16, pad=20)
        plt.tight_layout()
        
        return fig, ax


def main():
    """
    Main function to demonstrate the cellular Voronoi generator.
    """
    # Create generator with adjustable parameters
    generator = CellularVoronoi(
        width=800,
        height=600,
        num_cells=50,  # Fewer, larger cells
        wall_thickness_ratio=0.15,  # Thicker walls (15% of cell spacing)
        seed=42  # Set to None for random patterns
    )
    
    # Generate the pattern
    print("Generating cellular Voronoi pattern...")
    cells, vor = generator.generate()
    
    # Visualize
    print(f"Generated {len(cells)} cells")
    print(f"Average cell spacing: {generator.avg_spacing:.2f}")
    print(f"Wall thickness: {generator.wall_thickness:.2f}")
    
    fig, ax = generator.visualize(cells)
    
    # Save the figure
    plt.savefig('cellular_voronoi.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Pattern saved to cellular_voronoi.png")
    # plt.show()  # Commented out to avoid blocking


if __name__ == "__main__":
    main()