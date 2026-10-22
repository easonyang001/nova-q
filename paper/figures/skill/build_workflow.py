"""Editable diagram primitives shared by native Draw.io and SVG, no raster tracing.

The generated reference is a composition study, never a source of scientific claims.
Run from any directory. CairoSVG exports the independently authored SVG to PDF/PNG.
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import cairosvg

OUT = Path(__file__).resolve().parent
W, H = 1200, 500
NAVY, TEAL, AMBER, GRAY = '#17334a', '#087f8c', '#b36b12', '#647482'
svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg', width='7.16in', height='2.9833in', viewBox=f'0 0 {W} {H}')
ET.SubElement(svg, 'rect', width=str(W), height=str(H), fill='white')
model = ET.Element('mxGraphModel', dx=str(W), dy=str(H), grid='1', page='1', pageWidth=str(W), pageHeight=str(H))
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')
counter = 1

def cell(value, style, x, y, w, h):
    global counter
    counter += 1
    c = ET.SubElement(root, 'mxCell', id=str(counter), value=value, style=style, vertex='1', parent='1')
    ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as':'geometry'})
    return c

def text(x, y, value, size=18, color=NAVY, bold=False, anchor='start'):
    el = ET.SubElement(svg, 'text', x=str(x), y=str(y), fill=color, **{'font-family':'DejaVu Sans', 'font-size':str(size), 'text-anchor':anchor})
    if bold: el.set('font-weight', 'bold')
    el.text = value
    width = max(40, len(value)*size*.61)
    left = x-width/2 if anchor == 'middle' else x
    cell(value, f'text;html=0;strokeColor=none;fillColor=none;fontFamily=DejaVu Sans;fontSize={size};fontColor={color};fontStyle={int(bold)};align={"center" if anchor=="middle" else "left"};verticalAlign=middle;spacing=0;', left, y-size, width, size+5)

def shape(kind, x, y, w, h, stroke=GRAY, fill='none', sw=1.4, dashed=False):
    attr = dict(stroke=stroke, fill=fill, **{'stroke-width':str(sw)})
    if dashed: attr['stroke-dasharray']='5 4'
    if kind == 'ellipse': ET.SubElement(svg,'ellipse',cx=str(x+w/2),cy=str(y+h/2),rx=str(w/2),ry=str(h/2),**attr)
    else: ET.SubElement(svg,'rect',x=str(x),y=str(y),width=str(w),height=str(h),**attr)
    cell('', f'{kind};strokeColor={stroke};fillColor={fill};strokeWidth={sw};dashed={int(dashed)};', x,y,w,h)

def line(points, color=NAVY, sw=1.5, arrow=False, dashed=False):
    attr={'points':' '.join(f'{x},{y}' for x,y in points),'fill':'none','stroke':color,'stroke-width':str(sw)}
    if dashed: attr['stroke-dasharray']='5 4'
    ET.SubElement(svg,'polyline',**attr)
    global counter
    counter += 1
    c=ET.SubElement(root,'mxCell',id=str(counter),edge='1',parent='1',style=f'edgeStyle=none;rounded=0;html=0;strokeColor={color};strokeWidth={sw};endArrow={"block" if arrow else "none"};endFill=1;dashed={int(dashed)};')
    geo=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    for label,p in [('sourcePoint',points[0]),('targetPoint',points[-1])]: ET.SubElement(geo,'mxPoint',x=str(p[0]),y=str(p[1]),**{'as':label})
    if len(points)>2:
        arr=ET.SubElement(geo,'Array',**{'as':'points'})
        for x,y in points[1:-1]: ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
    if arrow:
        x,y=points[-1]; a,b=points[-2]; angle=math.atan2(y-b,x-a)
        p=[(x,y),(x-9*math.cos(angle)+4*math.sin(angle),y-9*math.sin(angle)-4*math.cos(angle)),(x-9*math.cos(angle)-4*math.sin(angle),y-9*math.sin(angle)+4*math.cos(angle))]
        ET.SubElement(svg,'polygon',points=' '.join(f'{a},{b}' for a,b in p),fill=color)

def triangle(x,y,color=AMBER,fill='white'):
    points=[(x,y-5),(x-5,y+4),(x+5,y+4),(x,y-5)]
    ET.SubElement(svg,'polygon',points=' '.join(f'{a},{b}' for a,b in points[:-1]),stroke=color,fill=fill,**{'stroke-width':'1.4'})
    cell('',f'triangle;direction=north;strokeColor={color};fillColor={fill};',x-5,y-5,10,9)

def table(x,y,labels,rows):
    widths=[64,54,60]
    shape('rectangle',x,y,sum(widths),76,stroke='#99a7b2',fill='white')
    shape('rectangle',x,y,sum(widths),24,stroke='none',fill='#edf3f6')
    for j,v in enumerate(labels): text(x+sum(widths[:j])+6,y+17,v,16,bold=True)
    for i,row in enumerate(rows):
        for j,v in enumerate(row): text(x+sum(widths[:j])+6,y+42+i*22,v,16)
    for k in (24,50):line([(x,y+k),(x+sum(widths),y+k)],'#cad3d9',.7)

SITES=[(31,63),(72,23),(113,53),(145,92)]
DEMAND=[(22,30),(51,81),(78,52),(113,91),(144,27),(158,63)]
def map_view(x,y,selected=False):
    shape('rectangle',x,y,180,112,stroke='#9cabb6',fill='#f7fafb')
    for pts in [[(0,91),(45,48),(88,57),(132,0)],[(0,20),(70,35),(120,84),(180,102)],[(35,112),(74,64),(143,61),(180,26)]]:
        line([(x+a,y+b) for a,b in pts],'#d4dde2',1)
    if selected:
        for idx,site in [(0,SITES[0]),(1,SITES[1]),(2,SITES[1]),(4,SITES[1])]:
            d=DEMAND[idx]; line([(x+site[0],y+site[1]),(x+d[0],y+d[1])],TEAL,.9)
    for i,(a,b) in enumerate(DEMAND): triangle(x+a,y+b,AMBER if not selected or i in [0,1,2,4] else GRAY,AMBER if selected and i in [0,1,2,4] else 'white')
    for i,(a,b) in enumerate(SITES):
        shape('ellipse',x+a-4,y+b-4,8,8,TEAL,TEAL if selected and i in [0,1] else 'white')
        if selected and i in [0,1]:shape('ellipse',x+a-8,y+b-8,16,16,TEAL,'none',1.5)

for i,label in enumerate(['1  Clarify','2  Model','3  Prepare','4  Solve','5  Deliver']):
    x=20+i*240
    text(x,30,label,23,bold=True)
    line([(x,43),(x+204,43)],TEAL,2)
    if i<4:line([(x+204,187),(x+229,187)],NAVY,1.8,True)

text(20,65,'Confirmed request',17,TEAL)
shape('rectangle',27,83,190,174,stroke='#b1bec8',fill='#fafcfd')
for y,v in [(110,'Select ≤ 2 sites.'),(139,'Maximize coverage.'),(178,'Public: sites, demand'),(207,'Fixed: budget, radius'),(236,'Output: selected IDs')]:text(38,y,v,16)
text(20,291,'Clarifier',18,bold=True)
text(20,316,'Facts + settings',16)

text(260,65,'Typed binary formulation',17,TEAL)
text(270,109,'xᵢ: site selected',19)
text(270,136,'yⱼ: demand covered',19)
text(270,174,'max  Σⱼ wⱼ yⱼ',23,bold=True)
text(270,206,'yⱼ ≤ Σᵢ aⱼᵢ xᵢ',21)
text(270,237,'Σᵢ xᵢ ≤ 2;  xᵢ, yⱼ ∈ {0,1}',17)
text(260,291,'Modeler',18,bold=True)
text(260,316,'Schema + builder',16)

text(500,65,'Public sources → instance',17,TEAL)
table(501,83,['site','lat','lon'],[['s₁','…','…'],['s₂','…','…']])
text(501,179,'Demand + radius → aⱼᵢ',16)
map_view(501,189)
text(500,328,'Data Scout + Adapter',18,bold=True)

text(740,65,'QUBO → binary candidate',17,TEAL)
text(748,103,'E = f + Σₖ λₖ Pₖ',22,bold=True)
nodes=[(771,155),(817,138),(865,159),(793,207),(849,212),(905,193)]
for a,b in [(0,1),(1,2),(0,3),(1,3),(1,4),(2,4),(2,5),(3,4),(4,5)]: line([nodes[a],nodes[b]],'#8da0ac',1)
for i,(x,y) in enumerate(nodes): shape('ellipse',x-7,y-7,14,14,TEAL,TEAL if i<3 else 'white')
text(748,249,'x = (1, 1, 0, 0)',20)
text(740,291,'Compiler + backend',18,bold=True)
text(740,316,'Preflight + bounded calls',16)

text(980,65,'Original-constraint checks',16,TEAL)
map_view(981,83,True)
shape('ellipse',986,215,9,9,TEAL,TEAL)
text(1006,224,'Selected site',16)
triangle(990,245,AMBER,AMBER)
text(1006,250,'Covered demand',16)
text(980,291,'Visualizer + explanation',15,bold=True)
text(980,316,'Decision + provenance',16)

# Distinct evidence paths: role labels intentionally state only implemented scope.
line([(27,248),(10,248),(10,358),(210,358),(210,370)],GRAY,1,True,True)
text(137,349,'Request + data schema',16,GRAY)
text(20,389,'Requirement Checker',18,bold=True)
text(20,412,'Separate feasibility / ranking evaluator',16)
line([(470,259),(470,362),(330,362),(330,370)],GRAY,1,True,True)
text(455,389,'Data Checker',18,bold=True)
text(455,412,'Independent counts / totals',16)
line([(595,165),(705,165),(705,359),(596,359)],GRAY,1,True,True)
text(857,389,'Delivery status',18,bold=True)
text(857,412,'Usable / limited / unavailable',16)
line([(1071,326),(1071,361)],GRAY,1,True)

line([(1100,441),(1100,470),(438,470),(438,326),(370,326)],AMBER,1.8,True)
line([(825,470),(825,341),(650,341)],AMBER,1.8,True)
line([(160,424),(160,441),(1100,441)],AMBER,1,False)
line([(610,424),(610,441)],AMBER,1)
shape('rectangle',432,449,543,25,stroke='none',fill='white')
text(445,468,'Validation evidence → bounded data or model repair',17,AMBER)
text(20,494,'Schematic coverage example; checkers provide evidence, not a correctness guarantee.',16,GRAY)

# Optional context is visually separated from the main data path.
text(270,263,'Optional pattern retrieval',16,GRAY)


ET.ElementTree(svg).write(OUT/'workflow-editable.svg',encoding='utf-8',xml_declaration=True)
mxfile=ET.Element('mxfile',host='app.diagrams.net')
diagram=ET.SubElement(mxfile,'diagram',name='Workflow and validation interfaces',id='workflow')
diagram.append(model)
ET.ElementTree(mxfile).write(OUT/'workflow-editable.drawio',encoding='utf-8',xml_declaration=True)
cairosvg.svg2pdf(url=str(OUT/'workflow-editable.svg'),write_to=str(OUT/'workflow-editable.pdf'))
cairosvg.svg2png(url=str(OUT/'workflow-editable.svg'),write_to=str(OUT/'workflow-editable.png'),output_width=2400,output_height=1000)
print(f'Generated editable Draw.io and SVG with PDF/PNG exports: {OUT}')
