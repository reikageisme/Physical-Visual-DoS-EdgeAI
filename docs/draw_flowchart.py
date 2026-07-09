import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

color_input = "#d1e7dd"
color_process = "#cfe2ff"
color_model = "#e2d9f3"
color_eval = "#fff3cd"
color_output = "#f8d7da"
color_edge = "#333333"

def draw_block(x, y, w, h, text, bg):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=1", 
                                 edgecolor=color_edge, facecolor=bg, lw=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=12, fontweight='bold', color='#111111')

def draw_arrow(x1, y1, x2, y2, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="black", lw=2.5, connectionstyle=f"arc3,rad={rad}"))

# Row 1 (y=75)
draw_block(5, 75, 25, 15, "1. Victim Feed\n(Real-world Input)", color_input)
draw_block(37.5, 75, 25, 15, "2. Saliency Extraction\n(Top 5% ROI)", color_process)
draw_block(70, 75, 25, 15, "7. NMS Bottleneck\n(O(N²) IoU Operations)", color_output)

# Row 2 (y=45)
draw_block(5, 45, 25, 15, "3. Genetic Algorithm\n(Patch Evolution)", color_process)
draw_block(37.5, 45, 25, 15, "4. Physical EOT\n(Rotate, Blur, Noise)", color_process)
draw_block(70, 45, 25, 15, "5. YOLOv8n (Gray-Box)\n(Edge Server Inference)", color_model)

# Row 3 (y=15)
draw_block(37.5, 15, 25, 15, "6. Fitness Evaluation\n(Max Boxes + Confidence)", color_eval)

# Output below row 1
draw_block(70, 95, 25, 12, "8. Visual DoS\n(CPU 100%, Frame Drop)", color_output)

# Arrows
draw_arrow(30, 82.5, 37.5, 82.5) # 1 -> 2
draw_arrow(50, 75, 50, 60) # 2 -> 4

draw_arrow(30, 52.5, 37.5, 52.5) # 3 -> 4
draw_arrow(62.5, 52.5, 70, 52.5) # 4 -> 5

draw_arrow(82.5, 60, 82.5, 75) # 5 -> 7
draw_arrow(82.5, 90, 82.5, 95) # 7 -> 8
draw_arrow(82.5, 45, 82.5, 22.5, rad=-0.5) # 5 -> 6 (Curve right)

draw_arrow(37.5, 22.5, 17.5, 45, rad=0.2) # 6 -> 3

plt.title("Sponge Patch: Attack Architecture & Pipeline", fontsize=18, fontweight='bold', pad=0)
plt.tight_layout()
os.makedirs('docs', exist_ok=True)
plt.savefig('docs/Sponge_GA_Flowchart.png', dpi=300, bbox_inches='tight')
print("Image saved to docs/Sponge_GA_Flowchart.png!")
