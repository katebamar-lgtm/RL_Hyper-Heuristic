"""
Visualisation graphique du réseau multimodal.

Permet de :
  - dessiner le réseau complet (nœuds, arcs par mode, lignes TC)
  - mettre en surbrillance un itinéraire (résultat d'un algo ou d'un test manuel)

Usage rapide :
    from model.visualization import draw_network
    draw_network(net, route=route, save_path="reseau.png")
"""

from __future__ import annotations
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .entities import Mode
from .network import MultimodalNetwork, Connection

MODE_STYLE = {
    Mode.WALK: dict(color="#999999", linestyle="dotted", linewidth=1.0, zorder=1),
    Mode.BIKE: dict(color="#2ca02c", linestyle="dashed", linewidth=1.2, zorder=2),
    Mode.CAR: dict(color="#7f7f7f", linestyle="solid", linewidth=1.2, zorder=2),
    Mode.BUS: dict(color="#1f77b4", linestyle="solid", linewidth=2.5, zorder=3),
    Mode.TRAM: dict(color="#d62728", linestyle="solid", linewidth=2.5, zorder=3),
    Mode.METRO: dict(color="#9467bd", linestyle="solid", linewidth=2.5, zorder=3),
}


def draw_network(net: MultimodalNetwork,
                  route: Optional[List[Connection]] = None,
                  show_free_modes: bool = True,
                  title: str = "Réseau multimodal",
                  save_path: Optional[str] = None,
                  show: bool = False,
                  figsize=(9, 7)):
    if not net._built:
        net.build_graph()

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Arcs libres (marche/vélo/véhicule) en fond, discrets
    if show_free_modes:
        drawn_free = set()
        for e in net.edges:
            a, b = net.nodes[e.origin].coord, net.nodes[e.destination].coord
            style = MODE_STYLE[e.mode]
            ax.plot([a.x, b.x], [a.y, b.y], alpha=0.5, **style)
            drawn_free.add(e.mode)

    # 2) Lignes de transport en commun, plus visibles, avec leur nom au milieu
    drawn_tc = set()
    for line in net.lines.values():
        style = MODE_STYLE[line.mode]
        for seg in line.segments:
            a = net.nodes[seg.origin_stop].coord
            b = net.nodes[seg.destination_stop].coord
            ax.plot([a.x, b.x], [a.y, b.y], **style)
        drawn_tc.add(line.mode)
        mid = line.stops[len(line.stops) // 2]
        mc = net.nodes[mid].coord
        ax.annotate(line.id, (mc.x, mc.y), fontsize=8, fontweight="bold",
                    color=style["color"], xytext=(4, 4), textcoords="offset points")

    # 3) Nœuds
    for node in net.nodes.values():
        c = node.coord
        if node.is_transit_stop:
            ax.scatter(c.x, c.y, s=90, facecolor="white", edgecolor="black", zorder=5)
        else:
            ax.scatter(c.x, c.y, s=45, facecolor="#dddddd", edgecolor="black", zorder=4)
        ax.annotate(node.id, (c.x, c.y), fontsize=8, xytext=(5, -8),
                    textcoords="offset points")

    # 4) Itinéraire mis en surbrillance (par-dessus tout le reste)
    if route:
        for i, conn in enumerate(route):
            a = net.nodes[conn.origin].coord
            b = net.nodes[conn.destination].coord
            ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y),
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        lw=2.8, alpha=0.85, shrinkA=8, shrinkB=8),
                        zorder=10)
        origin_c = net.nodes[route[0].origin].coord
        dest_c = net.nodes[route[-1].destination].coord
        ax.scatter(origin_c.x, origin_c.y, s=160, marker="o", facecolor="#2ca02c",
                   edgecolor="black", zorder=11, label="Origine")
        ax.scatter(dest_c.x, dest_c.y, s=160, marker="s", facecolor="#d62728",
                   edgecolor="black", zorder=11, label="Destination")

    # Légende des modes réellement présents
    legend_modes = drawn_tc | (set(MODE_STYLE) if show_free_modes else set())
    handles = [mpatches.Patch(color=MODE_STYLE[m]["color"], label=m.value)
               for m in Mode if m in legend_modes]
    if route:
        handles += [mpatches.Patch(color="#2ca02c", label="Origine"),
                    mpatches.Patch(color="#d62728", label="Destination")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return save_path
