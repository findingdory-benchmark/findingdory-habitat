import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

findingdory_receptacle_cats = {
    'explored area': 0, 
    'bathtub': 1,
    'bed': 2,
    'bench': 3,
    'cabinet': 4,
    'chair': 5,
    'chest_of_drawers': 6,
    'couch': 7,
    'counter': 8,
    'filing_cabinet': 9,
    'hamper': 10,
    'serving_cart': 11,
    'shelves': 12,
    'shoe_rack': 13,
    'sink': 14,
    'stand': 15,
    'stool': 16,
    'table': 17,
    'toilet': 18,
    'trunk': 19,
    'wardrobe': 20,
    'washer_dryer': 21,
    # 'unknown': 22,
}

# Define RGB colors for the legend items
colors = [
    # [255, 255, 255],  # navigable area
    [174, 199, 232],  # bathtub
    [255, 127, 14],   # bed
    [255, 187, 120],  # bench
    [44, 160, 44],    # cabinet
    [152, 223, 138],  # chair
    [214, 39, 40],    # chest_of_drawers
    [255, 152, 150],  # couch
    [148, 103, 189],  # counter
    [197, 176, 213],  # filing_cabinet
    [140, 86, 75],    # hamper
    [196, 156, 148],  # serving_cart
    [227, 119, 194],  # shelves
    [247, 182, 210],  # shoe_rack
    [127, 127, 127],  # sink
    [199, 199, 199],  # stand
    [188, 189, 34],   # stool
    [219, 219, 141],  # table
    [23, 190, 207],   # toilet
    [158, 218, 229],  # trunk
    [57, 59, 121],    # wardrobe
    [82, 84, 163],    # washer_dryer
    [107, 110, 207],  # unknown
]

# Normalize RGB values to [0, 1] range
normalized_colors = [[c/255 for c in color] for color in colors]

# Create patches for each category, ensuring correct alignment with colors
patches = [mpatches.Patch(color=normalized_colors[findingdory_receptacle_cats[cat]], label=f'{cat}') for cat in findingdory_receptacle_cats.keys()]

# Create the legend figure
fig, ax = plt.subplots(figsize=(10, 3))
legend = ax.legend(handles=patches, loc='center', fontsize='medium', ncol=4, frameon=False)
ax.axis('off')
plt.savefig('legend.png', bbox_inches='tight', pad_inches=0)
