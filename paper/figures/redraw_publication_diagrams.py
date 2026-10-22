"""Editable vector illustrations of implemented workflow mechanisms (not measured results)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Arc
from matplotlib.path import Path as MPath
from matplotlib.backends.backend_pdf import PdfPages
BASE=Path(__file__).resolve().parent
OUT=BASE/'vector';OUT.mkdir(exist_ok=True)
INK='#263B43';GRAY='#627178';LIGHT='#CDD6D8';TEAL='#176B71';GOLD='#A66A35';BLUE='#365D87'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.linewidth':.6,'text.color':INK,'mathtext.fontset':'stix'})

def canvas(height=3.6):
 fig,ax=plt.subplots(figsize=(7.16,height));fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.985)
 ax.set(xlim=(0,18),ylim=(0,height*2.4));ax.axis('off');return fig,ax

def txt(ax,x,y,s,size=8.4,color=INK,weight='normal',ha='left',va='center',**kw):
 return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.45,**kw)
def line(ax,a,b,color=LIGHT,lw=.8,style='-'):
 ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,ls=style,zorder=1)
def arrow(ax,a,b,color=GRAY,rad=0,lw=1,style='-'):
 p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,connectionstyle=f'arc3,rad={rad}',
  color=color,lw=lw,linestyle=style,shrinkA=0,shrinkB=0,zorder=1);ax.add_patch(p)
def dot(ax,x,y,color=TEAL,r=.065):ax.add_patch(Circle((x,y),r,fc=color,ec='none',zorder=3))
def badge(ax,x,y,n,color=TEAL,r=.21):
 ax.add_patch(Circle((x,y),r,fc='white',ec=color,lw=1));txt(ax,x,y,n,7.4,color,'bold',ha='center')
def person(ax,x,y):
 ax.add_patch(Circle((x,y+.11),.10,fill=False,ec=GOLD,lw=1))
 ax.add_patch(Arc((x,y-.13),.43,.32,theta1=0,theta2=180,color=GOLD,lw=1))
def document(ax,x,y,color=TEAL):
 v=np.array([[x-.3,y-.37],[x+.26,y-.37],[x+.26,y+.21],[x+.09,y+.4],[x-.3,y+.4],[x-.3,y-.37]])
 ax.add_patch(PathPatch(MPath(v),fill=False,ec=color,lw=1))
 for yy in [-.2,-.05,.1]:line(ax,(x-.17,y+yy),(x+.13,y+yy),color,.8)
 line(ax,(x+.09,y+.4),(x+.09,y+.21),color,.8);line(ax,(x+.09,y+.21),(x+.26,y+.21),color,.8)
def database(ax,x,y,color=TEAL):
 for d in [0,.22,.44]:ax.add_patch(Ellipse((x,y+.3-d),.75,.22,fill=False,ec=color,lw=.9))
 line(ax,(x-.375,y+.3),(x-.375,y-.3),color,.9);line(ax,(x+.375,y+.3),(x+.375,y-.3),color,.9)
 ax.add_patch(Arc((x,y-.3),.75,.22,theta1=180,theta2=360,ec=color,lw=.9))
def network(ax,x,y,color=TEAL):
 pts=[(x-.35,y+.25),(x+.34,y+.24),(x,y-.31)]
 for i,j in [(0,1),(1,2),(2,0)]:line(ax,pts[i],pts[j],color,.8)
 for a,b in pts:ax.add_patch(Circle((a,b),.085,fc='white',ec=color,lw=1))
def energy(ax,x,y,color=TEAL):
 z=np.linspace(-.43,.43,100);f=.3*np.cos(12*z)+1.2*z*z
 ax.plot(x+z,y+f-.22,color=color,lw=1.1);line(ax,(x-.48,y-.33),(x+.5,y-.33),GRAY,.6)
def outcome(ax,x,y,color=TEAL):
 for i,h in enumerate([.25,.48,.65]):line(ax,(x-.3+i*.22,y-.3),(x-.3+i*.22,y-.3+h),color,2)
 line(ax,(x-.43,y-.35),(x+.4,y-.35),GRAY,.7)
def route(ax,x,y,color=TEAL):
 pts=np.array([[-.37,.15],[-.12,.37],[.34,.23],[.27,-.28],[-.31,-.2],[-.37,.15]])
 ax.plot(x+pts[:,0],y+pts[:,1],color=color,lw=.9)
 for a,b in pts[:-1]:dot(ax,x+a,y+b,color,.04)
def coverage(ax,x,y,color=TEAL):
 ax.add_patch(Circle((x-.06,y),.33,fill=False,ec=color,lw=.8))
 for a,b in [(-.08,.02),(.22,.14),(-.25,-.12),(.45,-.28)]:dot(ax,x+a,y+b,color,.04)
def groups(ax,x,y,color=TEAL):
 for xx,yy in [(-.26,.15),(.23,.15),(0,-.24)]:
  for dx,dy in [(0,0),(.11,.03),(.03,.12)]:dot(ax,x+xx+dx,y+yy+dy,color,.035)
def medical(ax,x,y,color=TEAL):
 line(ax,(x-.26,y),(x+.26,y),color,1.6);line(ax,(x,y-.26),(x,y+.26),color,1.6)
 for xx,yy in [(-.42,.32),(.38,.3),(.4,-.26)]:dot(ax,x+xx,y+yy,color,.04)
def park(ax,x,y,color=TEAL):
 for dx in [-.22,.2]:
  ax.add_patch(Circle((x+dx,y+.13),.14,fill=False,ec=color,lw=.8))
  line(ax,(x+dx,y-.02),(x+dx,y-.28),color,.9)
 line(ax,(x-.38,y-.28),(x+.38,y-.28),GRAY,.7)
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=240,facecolor='white')
 return fig

# A. A narrative ribbon: decision roles above, implementation obligations below.
fig,ax=canvas(3.65)
xs=[2.0,5.5,9.0,12.5,16.0];y=5.02
line(ax,(.9,7.68),(17.1,7.68),LIGHT,.7)
txt(ax,.8,8.12,'Agent roles, executable checks and revision paths',8,GRAY,'bold')
for x,label in [(xs[0],'Confirm specification'),(xs[2],'Approve assumptions'),(xs[4],'Review delivery')]:
 person(ax,x,7.33);txt(ax,x,6.86,label,7.2,GOLD,ha='center')
for i,(x,fun,title,details) in enumerate(zip(xs,[document,network,database,energy,outcome],
 ['Clarify','Model','Prepare data','Anneal','Deliver'],
 ['Objectives, rules\nand assumptions','Binary decisions;\nobjective and rules',
  'Find and download;\njoin and transform','QUBO encoding;\nbounded solver calls','Decision, source,\nfigure and limitations'])):
 fun(ax,x,y);badge(ax,x-.86,y+.28,str(i+1));txt(ax,x,4.25,title,10.2,INK,'bold',ha='center')
 txt(ax,x,3.5,details,7.7,ha='center')
 if i<4:arrow(ax,(x+.61,y),(xs[i+1]-.6,y),GRAY,lw=.9)
for x,title,desc in zip(xs,['Source evidence','Model evidence','Data evidence','Original problem','Saved artifacts'],
 ['Fetched pages;\nflag unverified claims','Structural + semantic\nrequirement checks','IDs, units, periods;\nindependent aggregates','Feasibility and objective;\nconventional baseline','Model–data version;\nexplicit delivery status']):
 line(ax,(x,2.93),(x,2.63),LIGHT,.8);dot(ax,x,2.64,TEAL,.05)
 txt(ax,x,2.27,title,8,TEAL,'bold',ha='center');txt(ax,x,1.65,desc,7.2,GRAY,ha='center')
arrow(ax,(15.95,1.03),(5.45,1.03),GOLD,rad=-.045,lw=1.05)
arrow(ax,(8.85,1.03),(8.85,1.2),GOLD,lw=.8)
txt(ax,10.75,.57,'Evidence-driven repair • validate replacement • retain earlier usable version',7.3,GOLD,ha='center')
# Retrieval is a contextual annotation rather than a sixth processing stage.
arrow(ax,(5.5,6.23),(5.5,5.48),BLUE,style=(0,(2,2)),lw=.9)
txt(ax,5.5,6.47,'Optional modelling patterns',7.2,BLUE,ha='center')
save(fig,'workflow')

# B. Independent computations shown as branches and a reconvergence, not rows of boxes.
fig,ax=canvas(4.1)
txt(ax,.45,9.25,'Independent data and model computations',8,GRAY,'bold')
for row,source,producer,checker,metric,y in [
 (1,'Raw tables + request','Adapter','Data Checker','Counts and totals',6.87),
 (2,'Confirmed specification','Modeler','Requirement Checker','Feasibility and ranking',3.48)]:
 database(ax,1.25,y) if row==1 else document(ax,1.25,y)
 txt(ax,1.25,y-.78,'Raw tables\n+ request' if row==1 else 'Confirmed\nspecification',8.2,INK,'bold',ha='center')
 txt(ax,1.25,y-1.55,'Schema, units\nand derivations' if row==1 else 'Schema and\ndecision IDs',7.1,GRAY,ha='center')
 for py,title,ischeck in [(y+.62,producer,False),(y-.62,checker,True)]:
  txt(ax,4.7,py+.16,title,8.3,INK if not ischeck else TEAL,'bold')
  txt(ax,4.7,py-.32,'Transforms full tables' if row==1 and not ischeck else
      'Independent raw computation' if row==1 else 'Defines the binary model' if not ischeck else 'Evaluator + decision sampler',7.1,GRAY)
  arrow(ax,(1.85,y),(4.18,py),GRAY,rad=-.05 if not ischeck else .05,lw=.8)
  arrow(ax,(8.95,py),(9.13,y),TEAL if ischeck else GRAY,rad=.06 if not ischeck else -.06,lw=.85)
 ax.add_patch(Circle((9.42,y),.23,fill=False,ec=TEAL,lw=1.2));txt(ax,9.42,y,'=',10,TEAL,ha='center')
 arrow(ax,(9.71,y),(10.4,y),TEAL,lw=.9)
 txt(ax,10.65,y+.31,metric,9.2,TEAL,'bold')
 txt(ax,10.65,y-.32,'Compare prepared data with raw totals\nTolerance: 0.5% with unit-scale floor\nSkip counts for reducible sets' if row==1 else
     'Fix decisions; optimize helper variables\nTest valid, perturbed and delivered decisions',7.6)
 txt(ax,4.25,y-1.27,'Checker cannot see Adapter code' if row==1 else 'Checker cannot see model formulation',7.2,BLUE)
line(ax,(.45,1.17),(17.4,1.17),LIGHT,.8,style=(0,(3,3)))
txt(ax,.45,.74,'Evaluation only',7.7,GRAY,'bold')
txt(ax,4.06,.74,'Frozen raw-data truth',8.2,INK,'bold')
arrow(ax,(7.95,.74),(8.68,.74),GRAY,lw=.8)
txt(ax,9,.74,'Entity-level data  /  Model meaning  /  Delivered quality',8.1,INK)
txt(ax,.45,.18,'Separate roles can still share an LLM or schema error. Aggregate agreement is not a semantic guarantee.',7.3,GRAY)
save(fig,'verification_layers')

# C. Recovery as a branching evidence map with an explicit safe return path.
fig,ax=canvas(4.1)
txt(ax,.55,9.23,'Evidence and recovery decisions',8,GRAY,'bold')
# left-hand operational spine
ys=[7.8,5.8,3.8,1.8]
for y,n,title,details in zip(ys,'1234',
 ['Versioned artifacts','Check the instance and model','Preflight + solve','Evaluate and deliver'],
 ['Code + data + provenance','Data contracts and requirement evidence','HiGHS feasibility; bounded annealing','Original constraints, objective and source']):
 badge(ax,1.06,y,n,r=.23);txt(ax,1.58,y+.05,title,9.1,INK,'bold');txt(ax,1.58,y-.47,details,7.5,GRAY)
for a,b in zip(ys[:-1],ys[1:]):arrow(ax,(1.06,a-.35),(1.06,b+.34),GRAY,lw=.9)
# evidence branch and curved return
arrow(ax,(7.9,5.8),(9.05,6.55),GOLD,rad=.08)
network(ax,9.55,6.8,GOLD);txt(ax,10.15,6.95,'Repair data or model',9,GOLD,'bold')
txt(ax,10.15,6.36,'Schema mismatch → revisit modelling\nWrong filters or joins → revisit preparation',7.5)
arrow(ax,(9.55,7.3),(6.85,7.81),GOLD,rad=.38,lw=1.05)
txt(ax,12.5,8.24,'Validate before activation',7.6,GOLD,ha='center')
txt(ax,10.15,5.76,'Bounds and fixed user settings are protected.',7.1,GRAY)
# two separate solve outcomes
arrow(ax,(7.9,3.81),(9.02,4.1),GRAY,lw=.85)
dot(ax,9.42,4.13,GOLD,.1);txt(ax,9.85,4.35,'Proven infeasibility',8.9,GOLD,'bold')
txt(ax,9.85,3.79,'Return a diagnosis; do not submit to annealing.\nA timed-out preflight remains unknown.',7.5)
arrow(ax,(7.9,1.81),(9.03,2.0),TEAL,lw=.85)
outcome(ax,9.45,2.18);txt(ax,10.15,2.52,'Use a validated baseline',8.9,TEAL,'bold')
txt(ax,10.15,1.95,'If annealing is infeasible or materially worse,\nreport the conventional solution and its source.',7.5)
line(ax,(.55,.72),(17.4,.72),LIGHT,.8)
txt(ax,.55,.23,'A failed solver run alone does not justify changing the request.',7.6,INK)
txt(ax,17.4,.23,'Repairs and solver calls share a finite budget.',7.4,GRAY,ha='right')
save(fig,'recovery_routes')

# D. A sparse experimental design matrix: no colored table cells or UI-style panels.
fig,ax=canvas(4.7)
txt(ax,.5,10.8,'Experimental design and independent acceptance criteria',8,GRAY,'bold')
cols=[7.3,9.6,11.9,14.2,16.5]
for x,label in zip(cols,['Structured\nworkflow','Independent\ncheckers','Cross-stage\nrepair','Modelling\nretrieval','Clarification\nweb lookup']):
 txt(ax,x,9.98,label,7.6,INK,'bold',ha='center')
rows=['Direct LLM','Structured workflow','Workflow with checkers','Complete workflow','Workflow with retrieval','Without clarification lookup']
on=[[0,0,0,0,0],[1,0,0,0,1],[1,1,0,0,1],[1,1,1,0,1],[1,1,1,1,1],[1,1,1,0,0]]
for j,(name,state) in enumerate(zip(rows,on)):
 y=8.88-j*.69
 txt(ax,.55,y,name,8.2)
 for x,v in zip(cols,state):
  ax.add_patch(Circle((x,y),.085,fc=TEAL if v else 'white',ec=TEAL if v else LIGHT,lw=.7))
line(ax,(.55,4.75),(17.4,4.75),LIGHT,.8)
txt(ax,.55,4.33,'Every configuration searches for data; the final row disables only clarification lookup.',7.3,GRAY)
# The current long-form paper now covers five families with two scales; no run count is asserted.
txt(ax,.55,3.64,'Five development families, two scales each',7.8,GRAY,'bold')
for x,name,detail,fn in zip([1.65,5.25,8.85,12.45,16.05],
 ['Taipei chargers','YouBike tour','AED teams','London ambulance','Berlin playgrounds'],
 ['Population coverage','Closed inspection tour','Balanced partition','Facility assignment','New area coverage'],
 [coverage,route,groups,medical,park]):
 fn(ax,x,2.81);txt(ax,x,2.1,name,7.4,INK,'bold',ha='center');txt(ax,x,1.7,detail,6.9,GRAY,ha='center')
line(ax,(.55,1.2),(17.4,1.2),LIGHT,.8)
txt(ax,.55,.8,'External judge',8.3,TEAL,'bold')
txt(ax,5.2,.8,'Data correctness',8.1);txt(ax,10.2,.8,'Model correctness',8.1);txt(ax,14.5,.8,'Usable delivery',8.1)
txt(ax,.55,.24,'Frozen code, prompts and truth; independent outcomes; failures and unverifiable models retained.',7.3,GRAY)
save(fig,'acceptance_design')

with PdfPages(OUT/'publication_diagrams.pdf') as p:
 for number in plt.get_fignums():p.savefig(plt.figure(number))
print('Written: workflow, verification_layers, recovery_routes, acceptance_design (PDF / SVG / PNG).')
