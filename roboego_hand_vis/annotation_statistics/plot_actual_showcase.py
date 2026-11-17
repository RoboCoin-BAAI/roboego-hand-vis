"""Three presentation charts and one shared CSV. Measured L1/L3; synthetic L2."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, FancyBboxPatch
from matplotlib.path import Path as CurvePath
from .plot_showcase_samples import COLORS, INK, bar_panel


def export_chart_values(root, rows, planned=False):
    """One shareable CSV per figure; no cross-figure filtering required."""
    prefix = "ideal_" if planned else ""
    for kind in ["scenes", "actions", "objects"]:
        selected = [row for row in rows if row[0] == kind or
                    (kind == "actions" and row[0].startswith("actions_top6_"))]
        seen = set()
        with (root/(prefix+f"collection_plan_{kind}.csv")).open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.writer(handle)
            writer.writerow(["category","name","unit","L3","L2_estimated","L1"])
            for row in selected:
                key = (row[1], row[2])
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(row[1:4]+[format(v,".9g") for v in row[4:]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--basis",choices=["hours","files"],required=True)
    parser.add_argument("--planned",action="store_true",help="Label every figure as illustrative planned targets")
    parser.add_argument("--vocabulary-source",type=Path,help="Observed statistics used to extend planned vocabularies, scaled by dataset hours")
    args=parser.parse_args()
    raw=json.loads(args.source.read_text(encoding="utf-8"))
    if args.vocabulary_source:
        if not args.planned:
            parser.error("--vocabulary-source requires --planned")
        observed=json.loads(args.vocabulary_source.read_text(encoding="utf-8"))
        for name, dataset in raw["datasets"].items():
            reference=observed["datasets"][name]
            if reference["errors"] or reference["duration_s"] <= 0:
                raise ValueError(f"Invalid vocabulary source: {name}")
            projection=dataset["duration_s"]/reference["duration_s"]
            for field in ("actions", "objects"):
                for key, count in reference[field].items():
                    dataset[field].setdefault(key, count*projection)
    first=raw["datasets"]["guanglun_depth"];third=raw["datasets"]["L3_lerobot"]
    if first["errors"] or third["errors"]:
        raise ValueError("Source collection contains errors; inspect before plotting")
    measure="duration_s" if args.basis=="hours" else "files"
    scale=third[measure]/first[measure]
    root=args.output_root
    rows=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"font.size":10})
    note=("Projected coverage incl. unprocessed-data estimates; not verified inventory. " if args.planned else "L1 / L3 measured. ")+"L2: 75% L3 + 25% L1 by "+args.basis+"."
    prefix="ideal_" if args.planned else ""
    def save(fig,name):
        for text in list(fig.texts):
            if "LAYOUT SAMPLE" in text.get_text():text.remove()
        fig.text(.065,.022,note,fontsize=9,color="#6D7D90")
        fig.savefig(root/(prefix+name+".png"),dpi=180,facecolor="white")
        fig.savefig(root/(prefix+name+".pdf"),facecolor="white",bbox_inches="tight")
        plt.close(fig)
    # Scene hours, with one curly bracket for each category.
    a={(r["category"],r["area"]):r["hours"] for r in first["scenes"]}
    b={(r["category"],r["area"]):r["hours"] for r in third["scenes"]}
    scene={key:[a.get(key,0),3*b.get(key,0)+scale*a.get(key,0),b.get(key,0)] for key in a.keys()|b.keys()}
    cats=[c for c in ["household","commercial_service","industrial_production","professional_work","public_outdoor","leisure_social","other","unknown"] if any(k[0]==c for k in scene)]
    fig=plt.figure(figsize=(14,max(12,len(scene)*.34+4)),facecolor="white");ax=fig.add_axes([.34,.12,.59,.74])
    ys=[];names=[];values=[];groups=[];pos=0
    for cat in cats:
        start=pos
        for key in sorted([k for k in scene if k[0]==cat],key=lambda k:-sum(scene[k])):
            vals=scene[key];ys.append(pos);names.append(key[1].replace("_"," "));values.append(vals);pos+=1
            rows.append(["scenes",cat,key[1],"hours",vals[2],vals[1],vals[0]])
        groups.append((cat,start,pos-1));pos+=.8
    for i,color in enumerate(COLORS):ax.barh(ys,[v[i] for v in values],left=[sum(v[:i]) for v in values],height=.64,color=color,label="L"+str(i+1),edgecolor="white",linewidth=.35)
    ax.set_yticks(ys,names);ax.tick_params(axis="y",length=0,pad=9,labelsize=10);ax.invert_yaxis();ax.set_xlabel("Duration (hours)",fontsize=12,labelpad=10)
    ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    for cat,start,end in groups:
        top=start-.44;bottom=end+.44;mid=(top+bottom)/2;x=-.28;w=.023
        bend=min(.3,(bottom-top)/5)
        verts=[(x+w,top),(x,top),(x,top),(x,top+bend),(x,mid-bend),(x,mid-bend/2),(x-w,mid),(x,mid+bend/2),(x,mid+bend),(x,bottom-bend),(x,bottom),(x,bottom),(x+w,bottom)]
        ax.add_patch(PathPatch(CurvePath(verts,[CurvePath.MOVETO]+[CurvePath.CURVE4]*12),transform=ax.get_yaxis_transform(),fill=False,lw=1.4,color="#526479",clip_on=False))
        ax.text(x-.04,mid,cat.replace("_","\n"),transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=10.5,fontweight="bold",color=INK,clip_on=False)
    ax.legend(ncol=3,frameon=False,loc="upper right",bbox_to_anchor=(1,1.08))
    fig.text(.065,.95,("Projected " if args.planned else "")+"scene duration by hierarchy",fontsize=24,fontweight="bold",color=INK)
    l2_hours=(3*third["duration_s"]+scale*first["duration_s"])/3600
    kind="planned" if args.planned else "measured"
    fig.text(.065,.912,f'L1: {first["duration_s"]/3600:,.1f} h {kind} | L2: {l2_hours:,.1f} h estimated | L3: {third["duration_s"]/3600:,.1f} h {kind}',fontsize=11,color="#697B8F")
    save(fig,"collection_plan_scenes")
    for field,title,out in [("actions","Action vocabulary / top 50 verbs","collection_plan_actions"),("objects","Object inventory / top 50 concrete objects","collection_plan_objects")]:
        keys=set(first[field])|set(third[field])
        combined={key:[first[field].get(key,0),3*third[field].get(key,0)+scale*first[field].get(key,0),third[field].get(key,0)] for key in keys}
        all_names=sorted(combined,key=lambda k:(-sum(combined[k]),k))
        names=all_names[:50]
        counts=np.array([combined[k] for k in names]).T
        fig=plt.figure(figsize=(24,11),facecolor="white")
        bar_panel(fig,names,counts,("PLANNED / " if args.planned else "")+title,
                  "Illustrative target counts; not measured occurrences." if args.planned else "Counts from annotations; L2 is estimated from the specified mixture.")
        for key in all_names:
            v=combined[key];rows.append([field,"",key,"annotation occurrences",v[2],v[1],v[0]])
        if field=="actions":
            for col,level in enumerate([2,1,0]):
                card=fig.add_axes([.065+col*.31,.072,.285,.175]);card.axis("off")
                card.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=.06",transform=card.transAxes,facecolor=["#FFF7E7","#EAF8F5","#EDF5FF"][col],edgecolor="none"))
                card.text(.055,.81,"L"+str(level+1)+" / TOP 6",fontsize=12,fontweight="bold",color=INK)
                top=sorted(combined,key=lambda k:(-combined[k][level],k))[:6]
                for j,key in enumerate(top):
                    cx=.055+j%3*.32;cy=.48-j//3*.30
                    card.text(cx,cy,key,fontsize=11,fontweight="bold",color=INK)
                    v=combined[key][level];card.text(cx,cy-.14,f"{v:,.1f}" if level==1 else f"{v:,.0f}",fontsize=9,color="#6E7C8F")
                    if key not in names:
                        v=combined[key];rows.append(["actions_top6_L"+str(level+1),"",key,"annotation occurrences",v[2],v[1],v[0]])
        else:
            for x,key in [(.065,"cup"),(.53,"screwdriver")]:
                ax=fig.add_axes([x,.075,.435,.175]);ax.axis("off")
                ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=.04",transform=ax.transAxes,facecolor="#F1F5FA",edgecolor="none"))
                ax.text(.045,.78,key.upper(),fontsize=14,fontweight="bold",color=INK)
                ax.text(.045,.53,"Illustrative attribute\nexamples" if args.planned else "Observed attribute\nexamples",fontsize=9,color="#6D7D90",va="top",linespacing=1.6)
                attrs={}
                for source in [first,third]:
                    for attr,vs in source["attributes"].get(key,{}).items():
                        attrs.setdefault(attr,{})
                        for word,n in vs.items():attrs[attr][word]=attrs[attr].get(word,0)+n
                data=[]
                for attr in ["color","material","size"]:
                    vs=attrs.get(attr,{})
                    value=max(vs,key=vs.get) if vs else "not annotated"
                    if len(value)>38:value=value[:35]+"..."
                    data.append([attr.title(),value])
                table=ax.table(cellText=data,colWidths=[.23,.4],bbox=[.35,.12,.60,.75],cellLoc="left");table.auto_set_font_size(False);table.set_fontsize(9);table.set_zorder(5)
                for (r,c),cell in table.get_celld().items():cell.set_edgecolor("#E0E7F0");cell.set_linewidth(.6);cell.set_facecolor("white");cell.set_text_props(color=INK,weight="bold" if c==0 else "normal")
        save(fig,out)
    export_chart_values(root, rows, args.planned)
    print(note)
    print("Files:",first["files"],third["files"],"; L1 mixture scaling factor:",scale)
    print("Three independent CSVs exported:",len(rows),"plotted entries")


if __name__=="__main__":main()
