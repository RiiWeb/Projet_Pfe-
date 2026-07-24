import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

# Create figure with white background - horizontal layout
fig, ax = plt.subplots(figsize=(4, 1.2), dpi=96, facecolor='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.set_aspect('equal')
ax.axis('off')

# Shield icon (scaled down, positioned on left)
# Shift viewBox coordinates for smaller shield on left side
shield_x = [12, 3, 3, 12, 21, 21, 12]
shield_y = [2, 6.5, 12, 22, 12, 6.5, 2]

# Scale shield to fit in left portion
scale = 0.5
offset_x = 8
offset_y = 5

scaled_shield_x = [x * scale + offset_x for x in shield_x]
scaled_shield_y = [y * scale + offset_y for y in shield_y]

# Draw shield
shield = Polygon(list(zip(scaled_shield_x, scaled_shield_y)), 
                facecolor='#1a56db', alpha=0.12, 
                edgecolor='#1a56db', linewidth=0.8, 
                joinstyle='round')
ax.add_patch(shield)

# Draw checkmark (scaled)
check_x = [(9 * scale + offset_x), (11 * scale + offset_x), (13 * scale + offset_x)]
check_y = [(12 * scale + offset_y), (14 * scale + offset_y), (10 * scale + offset_y)]
ax.plot(check_x, check_y, color='#1a56db', linewidth=1.2, 
        solid_capstyle='round', solid_joinstyle='round')

# Add BLOCKSAFE text to the right
ax.text(20, 15, 'BLOCKSAFE', 
        fontsize=14, fontweight='bold', 
        color='#0d2a4a', 
        family='sans-serif',
        verticalalignment='center')

# Save with tight bounds
plt.savefig('c:\\pfe\\logo.png', dpi=96, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.3)
plt.close()
print("Logo PNG with text created successfully!")
