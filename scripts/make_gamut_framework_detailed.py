from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import image as mpimg

OUT=Path('/workspace/scratch/930c5f293dcb'); ASSET=Path('/workspace/shared/gamut_reference_assets')
INK='#0b0b0b'; INK2='#52514e'; BLUE='#2a78d6'; ORANGE='#eb6834'; AQUA='#1baf7a'; GRID='#e4e3df'
LB='#edf4fc'; LO='#fff1ea'; LA='#eaf8f2'; LG='#f6f6f4'; WHITE='#ffffff'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(21,10.5),facecolor=WHITE)
fig.text(.015,.982,'GAMUT: Class-Conditional Forecast Bands Under Type Uncertainty',fontsize=20,fontweight='bold',va='top')
fig.text(.985,.984,'one method · two fitting protocols',fontsize=9.5,color=INK2,ha='right',va='top')
axes=[fig.add_axes([.012,.105,.278,.815]),fig.add_axes([.305,.105,.410,.815]),fig.add_axes([.730,.105,.258,.815])]
for ax in axes:
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((.004,.004),.992,.992,boxstyle='round,pad=.004,rounding_size=.012',lw=.95,edgecolor=INK2,facecolor=WHITE))
def header(ax,t,s):
    ax.text(.03,.96,t,fontsize=12.5,fontweight='bold',va='top'); ax.text(.03,.924,s,fontsize=7.6,color=INK2,va='top'); ax.plot([.03,.97],[.9,.9],color=GRID,lw=.9)
def box(ax,x,y,w,h,t,s='',edge=INK2,face=WHITE,dashed=False,ts=8.2,ss=6.8,lw=1.15):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009,rounding_size=.012',lw=lw,linestyle='--' if dashed else '-',edgecolor=edge,facecolor=face,zorder=2))
    ax.text(x+w/2,y+h*.63 if s else y+h/2,t,ha='center',va='center',fontsize=ts,fontweight='bold',wrap=True,zorder=3)
    if s: ax.text(x+w/2,y+h*.26,s,ha='center',va='center',fontsize=ss,color=INK2,wrap=True,zorder=3)
def arrow(ax,a,b,c=INK,lw=1.1,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=lw,color=c,connectionstyle=f'arc3,rad={rad}',zorder=4))
def lab(ax,x,y,s,size=7.2,color=INK2,ha='left',weight='normal'):
    ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontweight=weight)
def chips(ax,x,y,w,h,names,colors,fs=6.2):
    gap=.008; cw=(w-gap*(len(names)-1))/len(names)
    for i,(n,c) in enumerate(zip(names,colors)):
        xx=x+i*(cw+gap); ax.add_patch(FancyBboxPatch((xx,y),cw,h,boxstyle='round,pad=.004,rounding_size=.006',lw=.75,edgecolor=c,facecolor=WHITE))
        ax.text(xx+cw/2,y+h/2,n,fontsize=fs,ha='center',va='center')
classes=['SN Ia','SN CC','SLSN','AGN','TDE','CV']; ccol=[BLUE,BLUE,ORANGE,AQUA,ORANGE,AQUA]

# A. Real motivation panel and protocol input.
ax=axes[0]; header(ax,'A. Observed prefix & type uncertainty','ZTF-BTS / FLARE folds')
im=mpimg.imread(ASSET/'observation.png'); ax.imshow(im,extent=[.055,.945,.665,.875],aspect='auto',zorder=1.5); ax.add_patch(Rectangle((.055,.665),.89,.21,fill=False,edgecolor=GRID,lw=.7,zorder=2))
lab(ax,.055,.642,'real observation study: class coverage and early misclassification',6.8)
box(ax,.05,.545,.27,.07,r'prefix features $x_0$','observed $g/r$ prefix',BLUE,LB)
box(ax,.365,.545,.27,.07,'FLARE classifier',r'raw posterior $\tilde p_c$',ORANGE,LO)
box(ax,.68,.545,.27,.07,'temperature scaling',r'calibrated $\hat p_c$',AQUA,LA)
arrow(ax,(.325,.58),(.355,.58),BLUE); arrow(ax,(.64,.58),(.67,.58),ORANGE)
lab(ax,.05,.505,'latent class is unknown at test time',7.0,INK,weight='bold'); chips(ax,.05,.435,.90,.052,classes,ccol,6.0)
lab(ax,.05,.408,'posterior mass is retained for every class hypothesis',6.8)
lab(ax,.05,.345,'forecast origin / future target',7.4,INK,weight='bold'); ax.plot([.07,.93],[.275,.275],color=INK2,lw=1)
for x,c in [(.36,ORANGE),(.61,BLUE),(.78,AQUA)]: ax.plot([x,x],[.258,.292],color=c,lw=1.3)
lab(ax,.07,.235,'prefix',6.7); lab(ax,.36,.235,'origin $h$',6.7,ORANGE,'center'); lab(ax,.78,.235,'future $g/r$ detections',6.7,AQUA,'center'); lab(ax,.93,.235,'$h<t\\leq90$ d',6.7,ha='right')
box(ax,.05,.105,.90,.075,'forecast task',r'class $z$ latent · horizon cell $b$ · target magnitude $y$',INK2,LG,ts=8.0,ss=6.6)

# B. Detailed method architecture.
ax=axes[1]; header(ax,'B. GAMUT calibration architecture','C1–C3 are one calibration method')
box(ax,.035,.755,.93,.105,'C1  Hypothesis-conditional forecaster',r'prefix $x_0$ + one-hot hypothesis $c$ $\rightarrow$ shared LightGBM quantile models',BLUE,LB,ts=9.2,ss=7.0)
chips(ax,.065,.685,.28,.042,classes,ccol,5.8); lab(ax,.355,.640,'all class hypotheses',6.8,BLUE,weight='bold')
for x,n in [(.555,'q16'),(.69,'median'),(.825,'q84')]: box(ax,x,.672,.115,.055,n,edge=BLUE,ts=6.9,lw=.8)
lab(ax,.945,.699,r'$\hat\mu_c,\hat\sigma_c$',6.6,ha='right'); arrow(ax,(.355,.615),(.545,.615),BLUE)
box(ax,.035,.575,.43,.08,r'standardized residuals $e=(y-\hat\mu_c)/\hat\sigma_c$','leave-one-object-out calibration scores',BLUE,WHITE,ts=7.0,ss=6.4)
box(ax,.505,.575,.46,.08,'class × lead-bin residual laws',r'$\hat f_{c,b}(y\mid x)$',BLUE,WHITE,ts=7.7,ss=7.0); arrow(ax,(.475,.615),(.495,.615),BLUE)
for i in range(6):
    for j in range(4): ax.add_patch(Rectangle((.545+i*.055,.592+j*.014),.043,.010,facecolor=LB if (i+j)%3==0 else WHITE,edgecolor=GRID,lw=.25))
box(ax,.035,.435,.42,.09,'C2  Calibrated membership',r'$\tilde p_c$ $\rightarrow$ temperature $T$ $\rightarrow$ $\hat p_c$',ORANGE,LO,ts=8.4,ss=7.0)
box(ax,.485,.435,.185,.09,'calibration objects','NLL / temperature fit',ORANGE,WHITE,ts=7.0,ss=6.2)
box(ax,.70,.435,.265,.09,'calibrated posterior',r'$\hat p_c(x_0)$',AQUA,LA,ts=7.7,ss=7.2); arrow(ax,(.67,.48),(.69,.48),ORANGE)
box(ax,.035,.235,.93,.135,'C3  Dual-weighted level-set calibration',r'$s_v(y\mid x)=\bar\sigma(x)\sum_c v_{b,c}\hat p_c\hat f_{c,b}(y\mid x)$; fixed point $\rightarrow\hat t_b$',AQUA,LA,ts=9.2,ss=7.0)
box(ax,.065,.145,.20,.05,'class CRC thresholds',r'$t_{c,b}(v)$',AQUA,WHITE,ts=6.8,ss=6.5,lw=.8)
box(ax,.30,.145,.20,.05,'merged horizon cells','lead-time bins',AQUA,WHITE,ts=6.8,ss=6.5,lw=.8)
box(ax,.535,.145,.20,.05,'fixed point',r'$v_c\leftarrow v_c(\bar t/t_c)^{1/2}$',AQUA,WHITE,ts=6.8,ss=6.3,lw=.8)
box(ax,.77,.145,.16,.05,'threshold',r'$\hat t_b=\min_c t_c$',AQUA,WHITE,ts=6.8,ss=6.3,lw=.8)
for a,b in [((.265,.17),(.29,.17)),((.50,.17),(.525,.17)),((.735,.17),(.76,.17))]: arrow(ax,a,b,AQUA)
arrow(ax,(.25,.575),(.25,.37),BLUE); arrow(ax,(.82,.435),(.82,.37),AQUA)
lab(ax,.035,.085,'default: cross-fitted empirical',7.0,BLUE,weight='bold'); lab(ax,.965,.085,'same GAMUT · strict-split certificate',7.0,AQUA,'right', 'bold'); ax.plot([.035,.965],[.06,.06],color=GRID,lw=.8)

# C. Output and real evidence.
ax=axes[2]; header(ax,'C. Forecast band & evaluation','real ZTF-BTS evidence')
im=mpimg.imread(ASSET/'day50.png'); ax.imshow(im,extent=[.055,.945,.695,.875],aspect='auto',zorder=1.5); ax.add_patch(Rectangle((.055,.695),.89,.18,fill=False,edgecolor=GRID,lw=.7,zorder=2)); lab(ax,.055,.672,'real day-50 measurements: FLARE vs. GAMUT',6.7)
box(ax,.05,.565,.90,.075,'GAMUT forecast band $C(x)$',r'$C(x)=\{y:s_v(y\mid x)\geq\hat t_b\}$ · may be a union of intervals',BLUE,LB,ts=8.8,ss=6.8); arrow(ax,(.5,.555),(.5,.53),BLUE)
box(ax,.05,.43,.42,.075,'coverage by class × horizon','nominal 68% / 95%',ORANGE,LO,ts=7.3,ss=6.4)
box(ax,.53,.43,.42,.075,'band width','relative to FLARE',AQUA,LA,ts=7.3,ss=6.4)
arrow(ax,(.5,.53),(.26,.51),ORANGE,rad=.12); arrow(ax,(.5,.53),(.74,.51),AQUA,rad=-.12)
box(ax,.05,.26,.90,.115,'same GAMUT · strict-split certification','disjoint train / calibration / test objects',AQUA,WHITE,True,8.0,6.6)
for x,w,t in [(.09,.14,'train'),(.31,.19,'A: nuisance + $v$'),(.57,.19,'B: thresholds'),(.81,.14,'test')]: box(ax,x,.292,w,.045,t,edge=AQUA,face=LA,ts=6.2,lw=.7)
for a,b in [((.235,.315),(.30,.315)),((.50,.315),(.56,.315)),((.76,.315),(.80,.315))]: arrow(ax,a,b,AQUA,lw=.9)
lab(ax,.50,.275,'finite-sample class × horizon certificate for eligible cells',6.5,ha='center')
im=mpimg.imread(ASSET/'ztf_classes.png'); ax.imshow(im,extent=[.055,.945,.055,.22],aspect='auto',zorder=1.5); ax.add_patch(Rectangle((.055,.055),.89,.165,fill=False,edgecolor=GRID,lw=.7,zorder=2)); lab(ax,.055,.035,'real ZTF-BTS evaluation: per-class coverage and band length',6.7)

fig.text(.296,.535,'→',fontsize=24,color=INK,ha='center',va='center'); fig.text(.719,.535,'→',fontsize=24,color=INK,ha='center',va='center')
fig.text(.012,.035,'Real-data insets: paper/figures/observation.pdf, day50.pdf, and ztf_classes.pdf; source plots from public ZTF-BTS results.',fontsize=7.2,color=INK2,ha='left',va='bottom')
fig.text(.988,.035,'Flat vector modules; source plots from the public benchmark.',fontsize=7.2,color=INK2,ha='right',va='bottom')
for ext in ('png','pdf','svg'): fig.savefig(OUT/f'GAMUT_framework_figure_detailed.{ext}',dpi=300,bbox_inches='tight',facecolor=WHITE)
plt.close(fig); print(OUT/'GAMUT_framework_figure_detailed.png')
