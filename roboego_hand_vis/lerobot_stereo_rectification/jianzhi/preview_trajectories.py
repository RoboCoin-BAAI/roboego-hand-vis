"""Plot head pose and MANO global translations without reading hand points."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    episode=args.source/'data/chunk-000/episode_000000'
    specs=[('Head','observation.head_pose',['x','y','z'],'#2468cc'),
           ('Left MANO translation','observation.hand_left_mano',['dim_0','dim_1','dim_2'],'#22a35a'),
           ('Right MANO translation','observation.hand_right_mano',['dim_0','dim_1','dim_2'],'#ed8624')]
    streams=[];summary={}
    for label,feature,cols,color in specs:
        table=pq.read_table(episode/(feature+'.parquet'))
        t=table['timestamp'].to_numpy();xyz=np.column_stack([table[c].to_numpy() for c in cols])
        valid=np.isfinite(t)&np.isfinite(xyz).all(axis=1)
        if not valid.any():raise ValueError('No finite samples: '+feature)
        t,xyz=t[valid],xyz[valid]
        streams.append((label,t,xyz,color))
        with (out/(feature+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(['timestamp','x','y','z']);writer.writerows(zip(t,*xyz.T))
        summary[feature]=dict(samples=len(t),source_columns=cols,start=xyz[0].tolist(),end=xyz[-1].tolist(),min=xyz.min(axis=0).tolist(),max=xyz.max(axis=0).tolist(),range=np.ptp(xyz,axis=0).tolist())
    fig=plt.figure(figsize=(17,10))
    for i,(label,t,xyz,color) in enumerate(streams):
        for row,relative in enumerate([False,True]):
            ax=fig.add_subplot(2,3,row*3+i+1,projection='3d')
            points=xyz-xyz[0] if relative else xyz
            stride=max(1,len(t)//1800)
            q=points[::stride];tt=t[::stride]
            ax.plot(*q.T,color=color,alpha=.35,lw=.6)
            scatter=ax.scatter(*q.T,c=tt,cmap='viridis',s=2)
            ax.scatter(*points[0],color='lime',edgecolors='black',s=65,marker='o',label='start')
            ax.scatter(*points[-1],color='red',s=70,marker='X',label='end')
            span=max(float(np.ptp(points,axis=0).max()),.01)
            mid=(points.max(axis=0)+points.min(axis=0))/2
            ax.set_xlim(mid[0]-span*.55,mid[0]+span*.55);ax.set_ylim(mid[1]-span*.55,mid[1]+span*.55);ax.set_zlim(mid[2]-span*.55,mid[2]+span*.55)
            ax.set_box_aspect((1,1,1));ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
            ax.set_title(label+(' | displacement from first sample' if relative else ' | stored coordinates'),fontsize=10)
            if i==0:ax.legend(fontsize=7)
    fig.colorbar(scatter,ax=fig.axes,shrink=.5,pad=.03,label='timestamp (s)')
    fig.suptitle('Head + MANO global_transl trajectories | no hand points, no axis changes\nCoordinate frames and translation origin are unconfirmed; panels are separate.',fontsize=13)
    fig.savefig(out/'trajectories_3d.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(17,10),sharex='col')
    for col,(label,t,xyz,color) in enumerate(streams):
        for row,axis in enumerate('XYZ'):
            ax=axes[row,col];ax.plot(t,xyz[:,row],color=color,lw=.7);ax.grid(alpha=.3);ax.set_ylabel(axis+' (stored units)')
            if row==0:ax.set_title(label)
            if row==2:ax.set_xlabel('timestamp (s)')
    fig.suptitle('Stored XYZ components vs time | independent streams; no resampling')
    fig.tight_layout();fig.savefig(out/'trajectories_xyz_time.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,13))
    for row,(label,t,xyz,color) in enumerate(streams):
        for col,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
            ax=axes[row,col];ax.scatter(xyz[::4,a],xyz[::4,b],c=t[::4],cmap='viridis',s=3)
            ax.scatter(xyz[0,a],xyz[0,b],c='lime',edgecolors='black',s=70);ax.scatter(xyz[-1,a],xyz[-1,b],c='red',marker='X',s=70)
            ax.set_xlabel('XYZ'[a]);ax.set_ylabel('XYZ'[b]);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.3);ax.set_title(label)
    fig.suptitle('XY / XZ / YZ projections | green start, red end, color = time');fig.tight_layout();fig.savefig(out/'trajectories_projections.png',dpi=150);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(note='MANO global_transl is a model translation, not confirmed anatomical wrist position. No common coordinate frame or units asserted.',streams=summary),indent=2),encoding='utf-8')
    print(json.dumps(summary));print(out)


if __name__=='__main__':main()
