"""Fixed presentation layouts. All counts and attributes here are illustrative."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as CurvePath

COLORS = ["#85BFFC", "#6CC8C5", "#F3CC81"]
INK = "#20374F"


def save(fig, root, png_name, pdf_name):
    fig.savefig(root / png_name, dpi=160, facecolor="white")
    fig.savefig(root / pdf_name, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def scene_sample(root):
    groups = [
        ("Household", ["Kitchen", "Dining area", "Living room", "Bedroom", "Bathroom"],
         [[450,120,35],[380,110,30],[420,130,40],[320,90,25],[290,100,30]]),
        ("Commercial", ["Retail store", "Restaurant", "Cafe", "Warehouse"],
         [[370,130,30],[330,110,35],[280,95,25],[400,140,40]]),
        ("Professional\nwork", ["Office", "Laboratory", "Classroom"],
         [[290,160,60],[250,180,80],[260,120,45]])]
    fig = plt.figure(figsize=(13,8), facecolor="white")
    ax = fig.add_axes([.33,.14,.61,.69])
    ys, names, counts, brackets = [], [], [], []
    pos = 0
    for group, areas, values in groups:
        start = pos
        for area, value in zip(areas, values):
            ys.append(pos); names.append(area); counts.append(value); pos += 1
        brackets.append((group,start,pos-1)); pos += .8
    for i, color in enumerate(COLORS):
        ax.barh(ys,[v[i] for v in counts],left=[sum(v[:i]) for v in counts],
                height=.64,color=color,label=f"L{i+1}",edgecolor="white",linewidth=.6)
    ax.set_yticks(ys,names); ax.tick_params(axis="y",length=0,pad=10,labelsize=11)
    ax.invert_yaxis(); ax.set_xlim(0,700); ax.set_xlabel("Duration (hours)",fontsize=12,labelpad=12)
    ax.grid(axis="x",alpha=.15); ax.set_axisbelow(True)
    for y, value in zip(ys,counts):
        ax.text(sum(value)+8,y,f"{sum(value)} h",va="center",fontsize=9,color="#5C6B7B")
    for label,start,end in brackets:
        top,bottom=start-.43,end+.43; mid=(top+bottom)/2; x=-.245; w=.025
        vertices=[(x+w,top),(x,top),(x,top),(x,top+.3),(x,mid-.25),(x,mid-.12),(x-w,mid),
                  (x,mid+.12),(x,mid+.25),(x,bottom-.3),(x,bottom),(x,bottom),(x+w,bottom)]
        ax.add_patch(PathPatch(CurvePath(vertices,[CurvePath.MOVETO]+[CurvePath.CURVE4]*12),
                     transform=ax.get_yaxis_transform(),fill=False,color="#526479",lw=1.6,clip_on=False))
        ax.text(x-.055,mid,label,transform=ax.get_yaxis_transform(),ha="right",va="center",
                fontsize=12,fontweight="bold",color=INK,clip_on=False)
    ax.legend(ncol=3,frameon=False,bbox_to_anchor=(1,1.14),loc="upper right",fontsize=11)
    fig.text(.075,.94,"Scene duration by hierarchy",fontsize=23,fontweight="bold",color=INK)
    fig.text(.075,.895,"Category braces group area rows; each bar stacks L1, L2 and L3 hours.",fontsize=11,color="#697B8F")
    fig.text(.075,.035,"LAYOUT SAMPLE / Illustrative values only; no measured data used.",fontsize=10,color="#7C8998")
    save(fig,root,"scene_layout_sample.png","collection_plan_scenes.pdf")


def bar_panel(fig, names, values, title, subtitle):
    order = np.argsort(-values.sum(axis=0),kind="stable")
    names = [names[i] for i in order]; values = values[:,order]
    ax = fig.add_axes([.065,.405,.91,.40])
    x = np.arange(len(names)); bottom = np.zeros(len(names))
    for i,color in enumerate(COLORS):
        ax.bar(x,values[i],bottom=bottom,width=.74,color=color,label=f"L{i+1}",linewidth=0)
        bottom += values[i]
    ax.set_xticks(x,names,rotation=52,ha="right",fontsize=9)
    ax.set_ylabel("Occurrences",fontsize=12,labelpad=10)
    ax.grid(axis="y",alpha=.16); ax.set_axisbelow(True)
    ax.ticklabel_format(axis="y",style="plain")
    ax.set_xlim(-.8,len(names)-.2)
    ax.legend(ncol=3,frameon=False,loc="upper right",bbox_to_anchor=(1,1.16))
    fig.text(.065,.94,title,fontsize=25,fontweight="bold",color=INK)
    fig.text(.065,.89,subtitle,fontsize=11,color="#697B8F")
    fig.text(.065,.025,"LAYOUT SAMPLE / Counts and attributes are illustrative, not measurements or collection commitments.",fontsize=9,color="#7C8998")
    return names,values


def action_sample(root):
    names = ["pick","place","grasp","move","wipe","open","close","lift","hold","release",
             "insert","pour","fold","push","pull","rotate","turn","cut","attach","remove",
             "screw","unscrew","tighten","align","write","draw","stir","wrap","peel","scrub"]
    values = np.array([
        [2200,2100,1050,1700,1800,1400,1300,1000,900,700,800,1100,1000,950,850,650,700,600,420,750,350,300,280,460,400,280,380,320,300,260],
        [750,790,920,680,500,530,500,700,500,550,820,400,500,400,430,620,420,550,580,510,600,550,490,520,390,310,300,370,260,280],
        [180,210,410,170,80,150,160,260,250,300,380,100,170,130,140,290,170,260,310,220,340,300,280,320,240,200,100,190,150,120]])
    fig = plt.figure(figsize=(16,9),facecolor="white")
    names,values = bar_panel(fig,names,values,"Action vocabulary / top 30 verbs",
        "One bar per verb; colored segments show occurrence counts at each level.")
    for column,level in enumerate([2,1,0]):
        x=.065+column*.31
        card=fig.add_axes([x,.072,.285,.175]); card.axis("off")
        card.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=.06",
                       transform=card.transAxes,facecolor=["#FFF7E7","#EAF8F5","#EDF5FF"][column],edgecolor="none"))
        card.text(.055,.81,f"L{level+1} / TOP 6",fontsize=12,fontweight="bold",color=INK)
        top = np.argsort(-values[level],kind="stable")[:6]
        for j,index in enumerate(top):
            cx=.055+(j%3)*.32; cy=.48-(j//3)*.30
            card.text(cx,cy,names[index],fontsize=12,fontweight="bold",color=INK)
            card.text(cx,cy-.14,f"{values[level,index]:,}",fontsize=9,color="#6E7C8F")
    save(fig,root,"action_layout_sample.png","collection_plan_actions.pdf")


def object_sample(root):
    names=["cup","bottle","box","tray","towel","cloth","pen","book","plate","bowl",
           "plastic bag","scissors","screwdriver","pliers","screw","nut","cable","keyboard","mouse",
           "brush","sponge","shirt","rope","tape","folder","paper","spoon","lid","comb","watering can"]
    values=np.array([
        [1500,1450,1350,1200,1100,1000,900,800,850,780,980,600,500,420,450,300,550,560,470,750,600,580,500,650,600,850,400,600,450,300],
        [650,600,580,520,430,380,450,390,300,280,400,480,640,590,650,520,560,470,410,370,300,310,380,420,350,400,260,360,200,180],
        [150,130,120,100,90,70,120,100,80,70,80,180,260,240,300,270,230,160,130,90,60,90,160,140,90,100,70,120,50,45]])
    fig=plt.figure(figsize=(16,9),facecolor="white")
    bar_panel(fig,names,values,"Object inventory / top 30 concrete objects",
        "Concrete names, not broad categories; synonyms and color variants share one label.")
    for x,title,rows in [(.065,"CUP",[["Color","white"],["Material","ceramic"],["Size","height 9 cm"]]),
                         (.53,"SCREWDRIVER",[["Color","red / black"],["Material","steel + polymer"],["Size","length 18 cm"]])]:
        ax=fig.add_axes([x,.075,.435,.175]);ax.axis("off")
        ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=.04",
                     transform=ax.transAxes,facecolor="#F1F5FA",edgecolor="none"))
        ax.text(.045,.78,title,fontsize=14,fontweight="bold",color=INK)
        ax.text(.045,.53,"Example object\nattributes",fontsize=10,color="#6D7D90",linespacing=1.6,va="top")
        table=ax.table(cellText=rows,colWidths=[.23,.40],bbox=[.35,.12,.60,.75],cellLoc="left")
        table.auto_set_font_size(False);table.set_fontsize(11)
        table.set_zorder(5)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor("#E0E7F0");cell.set_linewidth(.6);cell.set_facecolor("white")
            cell.set_text_props(color=INK,weight="bold" if c==0 else "normal")
    save(fig,root,"object_layout_sample.png","collection_plan_objects.pdf")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"font.size":10})
    scene_sample(args.output_root)
    action_sample(args.output_root)
    object_sample(args.output_root)
    print("Saved 3 single-page sample PDFs and their PNG previews. No input statistics read.")


if __name__=="__main__":
    main()
