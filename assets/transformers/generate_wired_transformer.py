"""Create a deliberately dense, original encoder-decoder wiring poster."""

import json

import matplotlib.pyplot as plt

from generate_diagrams import Diagram, OUT


def build():
    d = Diagram(
        "09_transformer_wiring",
        "TRANSFORMER / THE FULL WIRING VIEW",
        "Original-style post-norm encoder-decoder. Eight heads shown per attention module; dropout omitted at inference.",
        height=2500,
    )
    d.fig.set_size_inches(32, 25)
    d.ax.set_xlim(0, 3200)
    d.ax.texts[0].set_fontsize(32)
    d.elements[0].update(fontSize=52, width=2400, height=130)

    def node(x, y, w, label, color="blue", h=45, size=17):
        d.box(x, y, w, h, label, color, size=size)

    def mha(x, y, w, title, causal=False, cross=False):
        d.box(x, y, w, 530, "", "purple", container=True)
        d.text(x + 20, y + 15, title, size=24, bold=True)
        if cross:
            node(x + 25, y + 65, w / 2 - 40, "Q input: decoder features", "purple", size=18)
            node(x + w / 2 + 15, y + 65, w / 2 - 40, "K, V input: final encoder states", "green", size=18)
            q_source = x + w / 4
            kv_source = x + 3 * w / 4
        else:
            node(x + 25, y + 65, w - 50, "Current sequence representations X", "blue", size=18)
            q_source = kv_source = x + w / 2
        lane = (w - 40) / 8
        for i in range(8):
            left = x + 20 + i * lane
            center = left + lane / 2
            d.text(left + 15, y + 119, f"HEAD {i + 1}", size=14, bold=True)
            for j, letter in enumerate(["Q", "K", "V"]):
                bx = left + 5 + j * (lane - 10) / 3
                bw = (lane - 20) / 3
                node(bx, y + 155, bw, letter, "blue", h=38, size=15)
                source = q_source if letter == "Q" else kv_source
                d.arrow(source, y + 110, bx + bw / 2, y + 155, "gray")
                d.text(bx + 2, y + 198, f"W_{letter}{i+1}", size=11)
            node(left + 10, y + 235, lane - 20, "Q K^T", "orange", h=36, size=16)
            for j in range(2):
                d.arrow(left + 5 + j * (lane - 10) / 3 + (lane - 20) / 6,
                        y + 193, center, y + 235, "orange")
            mask = "causal + pad" if causal else "padding"
            node(left + 10, y + 295, lane - 20, f"/ sqrt(d_k)\n+ {mask} mask", "orange", h=55, size=12)
            d.arrow(center, y + 271, center, y + 295)
            node(left + 10, y + 375, lane - 20, "softmax", "purple", h=36, size=15)
            d.arrow(center, y + 350, center, y + 375)
            node(left + 10, y + 440, lane - 20, "Dropout(A) V", "green", h=36, size=13)
            d.arrow(center, y + 411, center, y + 440)
            vx = left + 5 + 2 * (lane - 10) / 3 + (lane - 20) / 6
            d.route([(vx, y + 193), (left + lane - 3, y + 215),
                     (left + lane - 3, y + 458), (left + lane - 10, y + 458)], "green")
            d.arrow(center, y + 476, x + w / 2, y + 495, "purple")
        node(x + 25, y + 495, w - 50, "Concat(head_1, ... , head_8)", "purple", h=32, size=16)
        return x + w / 2

    def output_projection(x, y, w):
        node(x, y, w, "Output projection W_O  ->  residual-branch dropout", "purple", size=18)
        d.arrow(x + w / 2, y + 45, x + w / 2, y + 80)
        node(x, y + 80, w, "Residual add + LayerNorm: LN(x + Sublayer(x))", "orange", size=19)

    def ffn(x, y, w):
        d.box(x, y, w, 150, "", "green", container=True)
        d.text(x + 20, y + 12, "POSITION-WISE FEED-FORWARD / shared across positions", size=20, bold=True)
        parts = [
            (x + 25, "Linear: d_model -> d_ff"),
            (x + w / 3 + 15, "ReLU"),
            (x + 2 * w / 3 + 5, "Linear: d_ff -> d_model"),
        ]
        for i, (bx, title) in enumerate(parts):
            node(bx, y + 65, w / 3 - 45, title, "green", h=50, size=16)
            if i < 2:
                d.arrow(bx + w / 3 - 45, y + 90, parts[i + 1][0], y + 90)

    d.text(65, 130, "ENCODER STACK  x N", size=32, bold=True)
    d.text(1685, 130, "DECODER STACK  x N", size=32, bold=True)
    for x, label in [(75, "Source text -> tokenizer -> source token IDs"),
                     (1695, "Shift-right target IDs / earlier generated tokens")]:
        node(x, 195, 1410, label, "blue", h=60, size=22)
        d.arrow(x + 705, 255, x + 705, 290)
        node(x, 290, 1410, "sqrt(d_model) * token embeddings + sinusoidal positional encoding -> dropout",
             "blue", h=60, size=19)
        d.arrow(x + 705, 350, x + 705, 455)

    mha(75, 390, 1410, "ENCODER MULTI-HEAD SELF-ATTENTION")
    d.arrow(780, 920, 780, 950)
    output_projection(100, 950, 1360)
    d.route([(780, 370), (45, 370), (45, 1052), (100, 1052)], "orange")
    d.arrow(780, 1075, 780, 1130)
    ffn(100, 1130, 1360)
    d.arrow(780, 1280, 780, 1320)
    node(100, 1320, 1360, "Dropout -> residual add -> LayerNorm", "orange", h=55, size=22)
    d.route([(780, 1100), (65, 1100), (65, 1347), (100, 1347)], "orange")
    d.arrow(780, 1375, 780, 1440)
    node(100, 1440, 1360, "ENCODER MEMORY E / final encoder output states [B, S, d_model]",
         "green", h=65, size=22)

    mha(1695, 390, 1410, "DECODER MASKED MULTI-HEAD SELF-ATTENTION", causal=True)
    d.arrow(2400, 920, 2400, 950)
    output_projection(1720, 950, 1360)
    d.route([(2400, 370), (3160, 370), (3160, 1052), (3080, 1052)], "orange")
    d.route([(2400, 1075), (1635, 1075), (1635, 1207), (1720, 1207)], "purple")

    mha(1695, 1120, 1410, "DECODER MULTI-HEAD CROSS-ATTENTION", cross=True)
    d.route([(1460, 1472), (1570, 1472), (1570, 1170),
             (2750, 1170), (2750, 1185)], "green")
    d.arrow(2400, 1650, 2400, 1680)
    output_projection(1720, 1680, 1360)
    d.route([(2400, 1100), (3180, 1100), (3180, 1782), (3080, 1782)], "orange")
    d.arrow(2400, 1805, 2400, 1840)
    ffn(1720, 1840, 1360)
    d.arrow(2400, 1990, 2400, 2030)
    node(1720, 2030, 1360, "Dropout -> residual add -> LayerNorm", "orange", h=55, size=22)
    d.route([(2400, 1820), (3150, 1820), (3150, 2057), (3080, 2057)], "orange")
    d.arrow(2400, 2085, 2400, 2120)
    node(1720, 2120, 1360, "Vocabulary projection: logits = H W_vocab + b  [B, T, |Vocab|]",
         "blue", h=55, size=22)
    d.arrow(2400, 2175, 2400, 2210)
    node(1720, 2210, 1360, "Softmax -> next-token distribution -> select token -> append -> repeat",
         "blue", h=55, size=21)

    d.box(75, 1580, 1410, 685, "", "purple", container=True)
    d.text(100, 1600, "TENSOR / PARAMETER / TRAINING CROSS-SECTION", size=25, bold=True)
    node(100, 1675, 1360,
         "Q = X W_Q ; K = X W_K ; V = X W_V ; d_k = d_model / h",
         "blue", h=65, size=23)
    node(100, 1770, 1360,
         "Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) + M) V",
         "purple", h=65, size=23)
    node(100, 1865, 1360,
         "Q: [B,h,T,d_k]   K: [B,h,S,d_k]   A: [B,h,T,S]   V: [B,h,S,d_v]",
         "orange", h=65, size=20)
    node(100, 1960, 1360,
         "LayerNorm(z) = gamma * (z - mean(z)) / sqrt(var(z) + epsilon) + beta",
         "orange", h=65, size=20)
    node(100, 2055, 1360,
         "Training: shifted targets -> next-token cross-entropy (ignore padding) -> backprop -> optimizer",
         "green", h=65, size=20)
    node(100, 2150, 1360,
         "Learned: embeddings, Q/K/V/O projections, FFN, norm scales/biases, vocabulary head",
         "green", h=65, size=18)
    d.text(85, 2310,
           "READING KEY: arrows inside attention heads are data flow; orange bypasses carry residual inputs.\n"
           "Bottom-left panel is a reference checklist, NOT a sequential computational pipeline. Batch dimension B is retained.\n"
           "One representative design, not every Transformer variant. Sinusoidal positions + ReLU + post-norm follow the original family.",
           size=22)
    for suffix in ["png", "svg"]:
        d.fig.savefig(OUT / f"09_transformer_wiring.{suffix}", facecolor="white")
    plt.close(d.fig)
    (OUT / "09_transformer_wiring.excalidraw").write_text(json.dumps({
        "type": "excalidraw", "version": 2, "source": "workshop",
        "elements": d.elements, "appState": {"viewBackgroundColor": "#ffffff"}, "files": {},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(d.elements)} editable elements: {OUT / '09_transformer_wiring.png'}")


if __name__ == "__main__":
    build()
