"""Skill-guided native Draw.io/SVG reconstructions of three composition studies."""
from pathlib import Path
import xml.etree.ElementTree as ET
import cairosvg

OUT=Path(__file__).resolve().parent
PRIMITIVES=(OUT/'build_workflow.py').read_text().split('SITES=')[0].replace('widths=[64,54,60]','widths=[68,66,75]')

def canvas(height):
    scope={ '__file__':str(OUT/'build_workflow.py') }
    exec(PRIMITIVES.replace('W, H = 1200, 500',f'W, H = 1200, {height}').replace("height='2.9833in'",f"height='{height*7.16/1200:.5f}in'"),scope)
    return scope

def export(c,name):
    svg=OUT/f'{name}.svg';ET.ElementTree(c['svg']).write(svg,encoding='utf-8',xml_declaration=True)
    mx=ET.Element('mxfile',host='app.diagrams.net');d=ET.SubElement(mx,'diagram',id=name,name=name.replace('-',' ').title());d.append(c['model'])
    ET.ElementTree(mx).write(OUT/f'{name}.drawio',encoding='utf-8',xml_declaration=True)
    cairosvg.svg2pdf(url=str(svg),write_to=str(OUT/f'{name}.pdf'))
    cairosvg.svg2png(url=str(svg),write_to=str(OUT/f'{name}.png'),output_width=2400)

def funcs(c):return [c[k] for k in ['text','line','shape','table','triangle']]

# Checker branches communicate input restrictions, not guaranteed independence.
c=canvas(620);t,l,s,table,tri=funcs(c)
for y,tag in [(30,'A  Data interpretation'),(243,'B  Requirement consistency'),(460,'C  External acceptance — evaluation only')]:
    t(20,y,tag,24,bold=True);l([(20,y+12),(1180,y+12)],'#b9c8d0',1)
table(25,63,['ID','value','unit'],[['v₁','120','people'],['v₂','80','people']])
t(25,164,'Raw-table profiles',18,bold=True);t(25,189,'Request + schema',18)
t(330,82,'Adapter',21,bold=True);t(330,109,'Code on full source tables',18)
t(330,161,'Data Checker',21,color=c['TEAL'],bold=True);t(330,188,'Separate count / total code',18)
l([(255,112),(278,112),(278,77),(315,77)],arrow=True)
l([(278,112),(278,157),(315,157)],arrow=True,color=c['TEAL'])
for y in [83,164]:l([(640,y),(710,y),(710,122),(750,122)],arrow=True)
s('ellipse',750,96,50,50,c['TEAL']);t(775,130,'≈',26,color=c['TEAL'],anchor='middle')
t(835,92,'Counts + numeric totals',21,bold=True)
t(835,120,'0.5% tolerance; unit-scale floor',18)
t(835,148,'Reducible-set counts skipped',18)
t(330,219,'Checker does not see Adapter code',18,color=c['GRAY'])

t(25,291,'Confirmed request',21,bold=True);t(25,320,'≤ 5 sites; count once',18)
t(25,350,'Schema + decision IDs',18)
t(330,289,'Modeler',21,bold=True);t(330,317,'Formulation + decision keys',18)
t(330,369,'Requirement Checker',21,color=c['TEAL'],bold=True);t(330,397,'Evaluator + decision sampler',18)
l([(270,322),(290,322),(290,287),(315,287)],arrow=True)
l([(290,322),(290,367),(315,367)],arrow=True,color=c['TEAL'])
for y in [291,373]:l([(642,y),(710,y),(710,332),(750,332)],arrow=True)
s('ellipse',750,306,50,50,c['TEAL']);t(775,340,'≈',26,color=c['TEAL'],anchor='middle')
t(835,297,'Feasibility + ranking',21,bold=True)
t(835,325,'Fix decisions; optimize helpers',18)
t(835,353,'Timeout / unsupported: inconclusive',17)
t(835,381,'Also evaluate delivered decisions',18)
t(330,429,'Checker does not see formulation',18,color=c['GRAY'])

t(25,510,'Frozen raw-data truth',21,bold=True);t(25,541,'Withheld from runtime agents',18)
l([(303,521),(351,521)],arrow=True)
t(366,510,'Case-specific judge',21,bold=True);t(366,541,'Prepared data, model, delivery',18)
l([(675,521),(722,521)],arrow=True)
for idx,(x,label) in enumerate([(760,'Data'),(895,'Model'),(1050,'Delivery')],1):
    s('ellipse',x-10,494,30,30,c['TEAL']);t(x+5,516,str(idx),19,color=c['TEAL'],anchor='middle');t(x+5,550,label,18,anchor='middle')
t(742,578,'Joint success requires all three gates',18,color=c['TEAL'])
t(20,610,'Runtime agreement is evidence; shared LLM/schema errors can remain. The judge is not a repair oracle.',17,color=c['GRAY'])
export(c,'verification-layers')

# Evidence routes, separated from solver-only search.
c=canvas(520);t,l,s,table,tri=funcs(c)
stages=[(20,'Versioned inputs','Model + data + sources'),(305,'Validate','Contracts + checkers'),(620,'Preflight / solve','HiGHS → bounded annealing'),(930,'Evaluate','Constraints + objective')]
for x,title,detail in stages:
    t(x,35,title,23,bold=True);t(x,63,detail,17)
    l([(x,76),(min(x+235,1180),76)],c['TEAL'],2)
for a,b in [(255,290),(550,605),(865,915)]:l([(a,53),(b,53)],arrow=True)
for x in [35,52,69]:s('rectangle',x,99,103,58,stroke=c['GRAY'],fill='white')
t(86,136,'vₜ',24,anchor='middle');t(20,188,'Fixed settings protected',18)

t(305,124,'Data mismatch',20,color=c['AMBER'],bold=True)
t(305,151,'→ revise preparation',18)
t(305,194,'Requirement mismatch',20,color=c['AMBER'],bold=True)
t(305,221,'→ revise formulation',18)
l([(555,124),(578,124),(578,270),(10,270),(10,140),(34,140)],color=c['AMBER'],arrow=True)
t(195,263,'Data / model replacement',18,color=c['AMBER'])
l([(555,194),(578,194)],color=c['AMBER'])

t(620,124,'Proven infeasible',20,bold=True);t(620,151,'No annealer submission',18)
t(620,194,'Timeout ≠ infeasibility',20,bold=True);t(620,221,'Status remains unknown',18)
t(620,266,'Feasible baseline',20,color=c['TEAL'],bold=True)
t(620,293,'Not always an optimum',17)
l([(869,124),(903,124),(903,364),(950,364)],arrow=True)

for y,title,detail in [(125,'Usable decision','Passes original requirements'),(245,'Validated baseline','Annealing fails / inferior'),(365,'Unavailable / limited','Report the unresolved gap')]:
    t(950,y,title,21,color=c['TEAL'] if y<365 else c['GRAY'],bold=True)
    t(950,y+29,detail,16)
l([(920,95),(920,117),(935,117)],arrow=True)
l([(868,269),(900,269),(900,237),(935,237)],arrow=True,color=c['TEAL'])

l([(20,320),(870,320)],'#b9c8d0',1)
t(20,350,'Before activating a replacement',22,bold=True)
for y,label in [(382,'Validate on original data; confirm changed bounds or formulas.'),(411,'Retain earlier usable version; tie every solution to its model/data version.'),(440,'Finite repair attempts, solver calls and process timeouts.')]:t(20,y,label,18)
t(20,496,'Solver failure alone does not establish a model error; penalty tuning remains a bounded inner loop.',17,color=c['GRAY'])
export(c,'recovery-routes')

# Evaluation protocol stays clearly prospective; no fabricated success-rate cells.
c=canvas(580);t,l,s,table,tri=funcs(c)
for x,label in [(20,'A  Development cases'),(405,'B  Mechanism ladder'),(860,'C  External judging')]:
    t(x,30,label,23,bold=True);l([(x,43),(min(x+320,1180),43)],c['TEAL'],2)
families=[('Taipei chargers','Maximum coverage'),('YouBike','Closed tour'),('Taipei AED','Balanced partition'),('London ambulance','p-median'),('Berlin playgrounds','Incremental coverage')]
for i,(name,kind) in enumerate(families):
    y=85+i*77;t(71,y,name,20,bold=True);t(71,y+26,kind+'; two scales',17)
    pts=[(25,y-9),(46,y-12),(35,y+11)]
    if i==1:
        l(pts+[pts[0]],c['TEAL'],1.4)
    elif i in [0,4]:s('ellipse',18,y-20,38,38,c['TEAL'])
    elif i==3:
        for p in pts[1:]:l([pts[0],p],c['TEAL'],1)
    for x,z in pts:s('ellipse',x-3,z-3,6,6,c['TEAL'],c['TEAL'])
t(20,495,'Fresh runs; no saved model or cache',17)

rows=[('Direct LLM','Whole-system baseline'),('Structured workflow','Typed interfaces + toolkit'),('Workflow with checkers','Add separate computations'),('Complete workflow','Add cross-stage repair'),('Workflow with retrieval','Add modelling patterns')]
for i,(name,note) in enumerate(rows):
    y=84+i*77
    s('ellipse',409,y-15,23,23,c['TEAL']);t(420,y+2,str(i+1),15,color=c['TEAL'],anchor='middle')
    t(448,y,name,20,bold=True);t(448,y+26,note,17)
    if 0<i<4:l([(421,y+14),(421,y+55)],c['TEAL'],1,True)
l([(405,438),(817,438)],c['GRAY'],1,dashed=True)
t(405,468,'Without clarification web lookup',19,bold=True)
t(405,495,'Acquisition still searches the web',17)

t(860,88,'Truth withheld from agents',18,bold=True)
for i,(name,note) in enumerate([('Data correct','Entities, coordinates, values'),('Model correct','Feasibility + objective ranking'),('Usable delivery','Requirements + reference quality')]):
    y=142+i*107;s('ellipse',864,y-18,29,29,c['TEAL']);t(878,y+3,str(i+1),18,color=c['TEAL'],anchor='middle');t(906,y,name,20,bold=True);t(860,y+32,note,17)
t(860,464,'Joint success = all three',20,color=c['TEAL'],bold=True)
t(860,495,'Three repetitions planned',17)
l([(20,525),(1180,525)],'#b9c8d0',1)
t(20,554,'Protocol, not completed results. Freeze code, prompts, cases, judge and truth; direct tools differ.',18,color=c['GRAY'])
export(c,'acceptance-design')
print('Rebuilt three native Draw.io/SVG diagrams and PDF/PNG previews.')
