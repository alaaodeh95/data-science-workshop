"""Regenerate the original workshop diagrams as PNG, SVG, and editable Excalidraw."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
COLORS = {
    "blue": ("#1864ab", "#d0ebff"),
    "green": ("#2f9e44", "#d3f9d8"),
    "purple": ("#862e9c", "#f3d9fa"),
    "orange": ("#e67700", "#fff3bf"),
    "gray": ("#495057", "#f1f3f5"),
}


class Diagram:
    def __init__(self, name, title, subtitle):
        self.name = name
        self.elements = []
        self.fig, self.ax = plt.subplots(figsize=(12, 6.6), dpi=160)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set(xlim=(0, 1200), ylim=(660, 0))
        self.ax.axis("off")
        self.text(45, 30, title, size=29, bold=True)
        self.text(45, 80, subtitle, size=17)

    def element(self, kind, **fields):
        item = {
            "id": f"{self.name}-{len(self.elements)}",
            "type": kind,
            "angle": 0,
            "strokeColor": "#000000",
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": None,
            "seed": 1,
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
            **fields,
        }
        self.elements.append(item)

    def text(self, x, y, text, size=18, bold=False):
        self.ax.text(
            x, y, text, fontsize=size * 0.6, va="top", color="black",
            family="DejaVu Sans", weight="bold" if bold else "normal",
            linespacing=1.5,
        )
        self.element(
            "text", x=x, y=y,
            width=max(len(line) for line in text.split("\n")) * size * 0.64,
            height=size * 2.5 * len(text.split("\n")),
            text=text, originalText=text, fontSize=size, fontFamily=2,
            textAlign="left", verticalAlign="top", containerId=None,
            lineHeight=1.25, autoResize=True,
        )

    def box(self, x, y, w, h, label, color="blue", size=20, container=False):
        stroke, fill = COLORS[color]
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
            facecolor="none" if container else fill, edgecolor=stroke, linewidth=1.5,
        ))
        self.element(
            "rectangle", x=x, y=y, width=w, height=h, strokeColor=stroke,
            backgroundColor="transparent" if container else fill, roundness={"type": 3},
        )
        if label:
            self.text(x + 16, y + 14, label, size=size)

    def arrow(self, x1, y1, x2, y2, color="gray"):
        stroke = COLORS[color][0]
        self.ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
            linewidth=1.6, color=stroke,
        ))
        self.element(
            "arrow", x=x1, y=y1, width=abs(x2-x1), height=abs(y2-y1),
            points=[[0, 0], [x2-x1, y2-y1]], strokeColor=stroke,
            startArrowhead=None, endArrowhead="arrow",
            startBinding=None, endBinding=None,
        )

    def save(self, takeaway):
        self.box(45, 565, 1110, 65, takeaway, "gray", size=19)
        self.fig.savefig(OUT / f"{self.name}.png", facecolor="white")
        self.fig.savefig(OUT / f"{self.name}.svg", facecolor="white")
        plt.close(self.fig)
        (OUT / f"{self.name}.excalidraw").write_text(json.dumps({
            "type": "excalidraw", "version": 2, "source": "workshop",
            "elements": self.elements, "appState": {"viewBackgroundColor": "#ffffff"},
            "files": {},
        }, indent=2) + "\n", encoding="utf-8")


def recurrence():
    d = Diagram("01_recurrence", "From passing a memory to looking things up",
                "Conceptual information paths, not measured attention or model output")
    d.text(45, 140, "LSTM: information travels through recurrent steps", bold=True)
    for i, label in enumerate(["The animal", "did not cross", "... because", "it was tired"]):
        x = 55 + 285 * i
        d.box(x, 205, 240, 70, label, "blue")
        if i < 3:
            d.arrow(x + 240, 240, x + 280, 240)
    d.text(65, 300, "Each step updates a carried state. Gates help retain information.", size=19)
    d.text(45, 345, "Self-attention: a position can directly read other allowed positions", bold=True)
    for i, label in enumerate(["animal", "street", "tired"]):
        x = 55 + i * 280
        d.box(x, 465, 200, 65, label, "green")
        d.arrow(925, 420, x + 100, 465, "purple")
    d.box(925, 390, 225, 85, '"it": query', "purple")
    d.save("Attention creates direct information paths; it does not guarantee correct understanding.")


def encoder_decoder():
    d = Diagram("02_encoder_decoder", "Encoder and decoder: read, then generate",
                "The roles existed before Transformers. Translation is one example.")
    d.text(45, 140, "Basic recurrent encoder-decoder: one fixed-size context", bold=True)
    d.box(45, 195, 240, 85, "I love learning\nInput sequence", "blue")
    d.box(330, 195, 190, 85, "Encoder\nReads input", "green")
    d.box(565, 195, 175, 85, "One context\nBottleneck", "orange")
    d.box(785, 195, 160, 85, "Decoder\nGenerates", "purple")
    d.box(990, 195, 170, 85, "J'aime ...\nOutput", "blue")
    for a, b in [(285, 330), (520, 565), (740, 785), (945, 990)]:
        d.arrow(a, 235, b, 235)
    d.text(45, 330, "With attention: keep every encoder position available", bold=True)
    for i, word in enumerate(["I", "love", "learning"]):
        y = 365 + i * 60
        d.box(65, y, 300, 45, f"{word}: state", "green", size=18)
        d.arrow(365, y + 22, 675, 420 + i * 30, "purple")
    d.box(675, 400, 480, 105, "Decoder step\nForms a weighted context", "purple")
    d.text(65, 540, "Decoder queries; encoder states provide keys and values.", size=16)
    d.save("The decoder can use different input information at each output step.")


def qkv():
    d = Diagram("03_attention", "Attention: match, normalize, mix",
                "Exactly the notebook's first experiment: Q = [3, 0], key dimension = 2")
    d.box(45, 145, 280, 80, "QUERY\nWhat do I need? [3, 0]", "purple")
    rows = [
        ("animal", "[2.0, 0.0]", "4.243", "0.954", "[1, 0, 0]"),
        ("street", "[0.4, 0.2]", "0.849", "0.032", "[0, 1, 0]"),
        ("tired", "[0.0, 2.0]", "0.000", "0.014", "[0, 0, 1]"),
    ]
    d.text(365, 160, "Scaled dot products -> softmax -> weighted values", bold=True)
    for x, heading in [(65, "POSITION"), (255, "KEY"), (490, "SCORE"), (700, "WEIGHT"), (925, "VALUE")]:
        d.text(x, 260, heading, size=18, bold=True)
    for i, row in enumerate(rows):
        y = 305 + i * 65
        for x, w, label, color in zip(
            [50, 240, 475, 685, 910], [175, 215, 190, 200, 245], row,
            ["blue", "blue", "orange", "purple", "green"],
        ):
            d.box(x, y, w, 50, label, color, size=18)
    d.text(65, 515, "Output = 0.954 x V(animal) + 0.032 x V(street) + 0.014 x V(tired)", size=20)
    d.save("Keys decide the match. Values supply the content. This is a blend, not a hard lookup.")


def mask():
    d = Diagram("04_causal_mask", "A causal mask hides the future",
                "Rows ask questions. Columns supply information. Diagonal positions remain visible.")
    words = ["I", "love", "learning", "today"]
    for offset, heading, causal in [(65, "Encoder: both directions", False), (660, "Decoder: current + earlier", True)]:
        d.text(offset, 150, heading, size=22, bold=True)
        for j, word in enumerate(words):
            d.text(offset + 95 + j * 96, 225, word, size=14)
        for i, word in enumerate(words):
            d.text(offset - 15, 285 + i * 57, word, size=14)
            for j in range(4):
                blocked = causal and j > i
                d.box(offset + 85 + j * 96, 270 + i * 57, 87, 49,
                      "BLOCK" if blocked else "read", "gray" if blocked else "green", size=15)
    d.text(75, 515, "At 'love', the decoder may read 'I' and 'love', but not 'learning' or 'today'.", size=20)
    d.save("Mask future scores with -infinity BEFORE softmax, so their attention weights become zero.")


def block():
    d = Diagram("05_decoder_block", "Inside a decoder-only Transformer",
                "A simplified GPT-2-style pre-normalization block; not the only possible arrangement")
    stages = [
        (45, "Token IDs\nPosition IDs", "blue"),
        (265, "Token + position\nembeddings", "blue"),
        (485, "Repeated\nblocks", "purple"),
        (705, "Final norm\n+ LM head", "orange"),
        (925, "Next-token\nscores", "green"),
    ]
    for x, label, color in stages:
        d.box(x, 150, 195, 85, label, color)
        if x < 925:
            d.arrow(x + 195, 192, x + 215, 192)
    d.box(45, 290, 1110, 240, "", "purple", container=True)
    d.text(65, 305, "One block, repeated with its own learned weights:", bold=True)
    chain = [
        (75, "Normalize", "gray"), (295, "Masked\nmulti-head\nattention", "purple"),
        (515, "Add block\ninput", "orange"), (735, "Normalize\nthen MLP", "green"),
        (955, "Add state\nfrom first\nresidual", "orange"),
    ]
    for x, label, color in chain:
        d.box(x, 370, 175, 100, label, color, size=19)
        if x < 955:
            d.arrow(x + 175, 420, x + 215, 420)
    d.text(65, 495, "Attention mixes positions. The MLP transforms features at each position.", size=19)
    d.save("Residual additions preserve the sublayer input; heads learn different projections, not fixed roles.")


def generation():
    d = Diagram("06_generation", "Text generation is a repeated prediction loop",
                "Illustrative token choices and probabilities, NOT captured DistilGPT2 output")
    for x, label, color in [
        (45, "Current context\nI like", "blue"),
        (335, "Decoder model\nScores vocabulary", "purple"),
        (625, "Softmax\nA distribution", "orange"),
        (915, "Choose a token\nAppend it", "green"),
    ]:
        d.box(x, 170, 240, 100, label, color)
        if x < 915:
            d.arrow(x + 240, 220, x + 285, 220)
    d.text(75, 335, "Example step", bold=True)
    d.box(75, 385, 300, 110, "' coffee'  0.40\n' tea'       0.25\nOthers     0.35", "orange", size=21)
    d.arrow(390, 440, 505, 440)
    d.box(525, 385, 260, 110, "Greedy choice:\n' coffee'", "green", size=21)
    d.arrow(800, 440, 890, 440)
    d.box(915, 385, 240, 110, "New context:\nI like coffee\nThen repeat", "blue", size=21)
    d.save("Training can score many positions in parallel; new text is still generated token by token.")


if __name__ == "__main__":
    for build in [recurrence, encoder_decoder, qkv, mask, block, generation]:
        build()
    print(f"Generated six original diagrams in {OUT}")
