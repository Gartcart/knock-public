import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import unicodedata
import time
import numpy as np


def normalize_text(text):
    """
    Normalize text to handle special characters
    """
    # Replace smart quotes and other special characters
    text = text.replace('\x92', "'")
    text = text.replace('\x93', '"')
    text = text.replace('\x94', '"')

    # Normalize to ASCII
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def print_progress(current, total, description="Progress", interval=10):
    """Simple progress printer that doesn't require external libraries"""
    if current == 0 or current == total or current % interval == 0:
        percentage = (current / total) * 100
        print(f"{description}: {current}/{total} ({percentage:.1f}%)")


def animate_matching(num_frames=25):
    """
    Create a smoother animated visualization of the graph building and matching process
    with improved transitions that show the decision-making process

    Parameters:
        num_frames: Number of frames to include in the animation
    """
    print("Starting the animation process...")
    start_time = time.time()

    # Read the CSV file with explicit error handling
    try:
        print("Reading CSV file...")
        df = pd.read_csv('data.csv', encoding='ISO-8859-1', header=None)
        print(f"CSV loaded successfully with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Use a dictionary to efficiently track unique pairs and their weights
    pair_data = defaultdict(dict)

    # Process the data and clean text
    print("Processing data and building pairs...")
    rows = list(df.iterrows())
    total_rows = len(rows)

    for i, (index, row) in enumerate(rows):
        # Print progress
        print_progress(i + 1, total_rows, "Processing rows")

        if index == 0:  # Skip header row
            continue

        # Clean up special characters in person names
        person1 = normalize_text(str(row[0]))
        person2 = normalize_text(str(row[1]))

        # Ensure consistent ordering of pairs
        person1, person2 = sorted([person1, person2])
        pair = (person1, person2)

        similarity = float(row[2])
        days = normalize_text(str(row[4]))

        # If this pair already exists in our dictionary
        if 'weight' in pair_data[pair]:
            # Store the minimum similarity score as per original logic
            pair_data[pair]['weight'] = min(pair_data[pair]['weight'], similarity)
        else:
            pair_data[pair] = {'weight': similarity, 'days': days}

    # Create the full graph
    print("Building graph...")
    G_full = nx.Graph()

    # Convert to list for progress reporting
    pairs = list(pair_data.items())
    total_pairs = len(pairs)

    # This will store our edges in order of weight (highest first)
    sorted_edges = []

    for i, ((person1, person2), data) in enumerate(pairs):
        # Print progress
        print_progress(i + 1, total_pairs, "Adding edges")

        weight = data['weight']
        G_full.add_edge(person1, person2, weight=weight, days=data['days'])
        sorted_edges.append((person1, person2, weight))

    # Sort edges by absolute weight value (descending)
    sorted_edges.sort(key=lambda x: abs(x[2]), reverse=True)

    # Find maximum weight matching for the full graph
    print("Finding maximum weight matching...")
    matching = nx.max_weight_matching(G_full, maxcardinality=False)
    matching_edges = list(matching)
    print(f"Found {len(matching_edges)} matched pairs.")

    # Get all unique nodes
    all_nodes = set(G_full.nodes())

    # Set up the figure with better aesthetics
    plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(12, 9), dpi=100)

    # Set a more muted background color
    fig.set_facecolor('#F8F8F8')
    ax.set_facecolor('#F8F8F8')

    # Create the graph with all nodes
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    # Create a more spaced layout for better visibility
    pos = nx.spring_layout(G_full, seed=42, k=2.0)

    # Divide our animation into distinct phases, allocating frames to each phase
    # 10% - Setup with nodes only
    # 40% - Adding edges (prioritize by weight)
    # 50% - Finding and highlighting the matching

    setup_frames = max(1, int(num_frames * 0.1))
    edge_frames = max(1, int(num_frames * 0.4))
    matching_frames = num_frames - setup_frames - edge_frames

    print(
        f"Animation phases: {setup_frames} setup frames, {edge_frames} edge frames, {matching_frames} matching frames")

    # Map our logical phases to frame numbers for the animation
    frame_to_phase = {}

    # Setup phase
    for i in range(setup_frames):
        frame_to_phase[i] = ("setup", i, setup_frames)

    # Edge adding phase - adding edges in order of weight significance
    edge_samples = np.linspace(0, len(sorted_edges) - 1, edge_frames, dtype=int)
    for i, edge_idx in enumerate(edge_samples):
        frame_num = setup_frames + i
        frame_to_phase[frame_num] = ("edge", edge_idx, len(sorted_edges))

    # Matching phase - gradually build up the matching
    matching_samples = np.linspace(0, len(matching_edges), matching_frames, dtype=int)
    for i, match_count in enumerate(matching_samples):
        frame_num = setup_frames + edge_frames + i
        frame_to_phase[frame_num] = ("matching", match_count, len(matching_edges))

    # Calculate edge color map - this will help visualize which edges are most likely
    # to be included in the matching
    edge_weights = [G_full[u][v]['weight'] for u, v in G_full.edges()]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    weight_range = max_weight - min_weight

    def get_edge_color(weight):
        if weight_range == 0:
            normalized = 0.5
        else:
            normalized = (weight - min_weight) / weight_range
        # Softer, less harsh color palette
        if weight < 0:
            # Soft blue for negative weights
            return (0.6, 0.7, 0.9)
        else:
            # Use a more muted color spectrum for positive weights
            if normalized < 0.33:
                # Soft teal
                return (0.5, 0.8, 0.8)
            elif normalized < 0.67:
                # Muted gold
                return (0.9, 0.8, 0.5)
            else:
                # Soft coral
                return (0.9, 0.6, 0.5)

    def init():
        """Initialize the empty graph"""
        ax.clear()
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        ax.set_title("Building Graph and Finding Maximum Weight Matching")
        ax.axis('off')
        return []

    def update(frame_idx):
        """Update function with improved visualization of decision process"""
        ax.clear()

        # Get the phase, index, and total for this frame
        phase, idx, total = frame_to_phase[frame_idx]

        # Current state of the graph for this frame
        current_G = nx.Graph()
        current_G.add_nodes_from(all_nodes)

        # Title and visualization depends on the phase
        if phase == "setup":
            progress = (idx + 1) / total * 100
            ax.set_title(f"Setting up graph ({progress:.0f}%)")
            # Just show nodes in setup phase

        elif phase == "edge":
            # Add edges up to this point in order of weight significance
            for i in range(min(idx + 1, len(sorted_edges))):
                u, v, w = sorted_edges[i]
                current_G.add_edge(u, v, weight=w)

            if idx < len(sorted_edges):
                u, v, w = sorted_edges[idx]
                ax.set_title(f"Adding significant edges: {u} -- {v} (weight: {w:.2f})")
            else:
                ax.set_title("Complete Graph")

        elif phase == "matching":
            # Add all edges to the graph
            for u, v, w in sorted_edges:
                current_G.add_edge(u, v, weight=w)

            # In matching phase, show the decision process
            ax.set_title(f"Finding Maximum Weight Matching: {idx}/{total} edges selected")

        # Draw nodes
        nx.draw_networkx_nodes(current_G, pos, node_size=300, node_color="lightblue", ax=ax)

        # Draw edges with color based on weight
        if phase == "edge" or phase == "matching":
            # Create a list of edge colors based on weights
            edge_colors = []
            edge_widths = []

            for u, v in current_G.edges():
                weight = G_full[u][v]['weight']
                edge_colors.append(get_edge_color(weight))
                # Width proportional to absolute weight value
                edge_widths.append(0.5 + min(abs(weight) / 5, 3.0))

            nx.draw_networkx_edges(current_G, pos, width=edge_widths, edge_color=edge_colors, ax=ax)

            # Add edge labels showing weights for significant edges
            if len(current_G.edges()) <= 20:  # Only show labels if not too crowded
                edge_labels = {(u, v): f"{G_full[u][v]['weight']:.1f}"
                               for u, v in current_G.edges()}
                nx.draw_networkx_edge_labels(current_G, pos, edge_labels=edge_labels,
                                             font_size=7, ax=ax)

            # Highlight the matching edges
            if phase == "matching" and idx > 0:
                visible_matching = list(matching_edges)[:idx]
                # Highlight the matching edges with a softer distinct color
                nx.draw_networkx_edges(current_G, pos, edgelist=visible_matching,
                                       width=4.0, edge_color="#AA6DAA", style='solid', ax=ax)

        # Draw labels
        nx.draw_networkx_labels(current_G, pos, font_size=8, ax=ax)

        # Add a legend explaining the colors
        if phase == "edge" or phase == "matching":
            # Add custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=(0.6, 0.7, 0.9), lw=2, label='Negative Weight'),
                Line2D([0], [0], color=(0.5, 0.8, 0.8), lw=2, label='Low Positive Weight'),
                Line2D([0], [0], color=(0.9, 0.8, 0.5), lw=2, label='Medium Positive Weight'),
                Line2D([0], [0], color=(0.9, 0.6, 0.5), lw=2, label='High Positive Weight'),
                Line2D([0], [0], color="#AA6DAA", lw=3, label='Selected Match')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax.axis('off')
        return []

    # Create animation with smoother transition
    print(f"Creating animation with {num_frames} frames...")
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames,
        init_func=init, blit=False, interval=1000, repeat=True
    )

    # Save animation with higher quality
    print("Saving animation...")
    try:
        print("Attempting to save as GIF...")
        ani.save('graph_matching_smooth.gif', writer='pillow', fps=1, dpi=100)
        print("Animation saved as 'graph_matching_smooth.gif'")
    except Exception as e:
        print(f"Error saving as GIF: {e}")
        try:
            print("Trying to save as MP4...")
            ani.save('graph_matching_smooth.mp4', writer='ffmpeg', fps=5, dpi=100,
                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            print("Animation saved as 'graph_matching_smooth.mp4'")
        except Exception as e2:
            print(f"Error saving as MP4: {e2}")
            print("Saving individual frames instead...")

            # Save individual frames as a last resort
            import os
            frames_dir = "animation_frames"
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)

            for i in range(num_frames):
                # Update the figure for this frame
                update(i)
                # Save the frame
                plt.savefig(f"{frames_dir}/frame_{i:04d}.png", dpi=100)
                print(f"Saved frame {i + 1}/{num_frames}")

    # Create a final static visualization
    plt.figure(figsize=(12, 10))

    # Draw the full graph with weighted edges
    edge_colors = [get_edge_color(G_full[u][v]['weight']) for u, v in G_full.edges()]
    edge_widths = [0.5 + min(abs(G_full[u][v]['weight']) / 5, 2.5) for u, v in G_full.edges()]

    nx.draw(G_full, pos, node_color="lightblue", edge_color=edge_colors,
            width=edge_widths, with_labels=True, node_size=400, font_size=8)

    # Highlight the matching with a soft purple
    nx.draw_networkx_edges(G_full, pos, edgelist=matching_edges,
                           width=4.0, edge_color="#AA6DAA")

    plt.title("Final Maximum Weight Matching")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("final_matching_improved.png", dpi=150)
    print("Final matching visualization saved as 'final_matching_improved.png'")

    # Show the animation
    plt.show()

    # Calculate and display total time
    total_time = time.time() - start_time
    print(f"Total process completed in {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")

    return G_full, matching


# Run the animation
if __name__ == "__main__":
    animate_matching(num_frames=25)  # Use exactly 25 frames