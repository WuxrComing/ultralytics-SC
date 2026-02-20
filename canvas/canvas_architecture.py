"""
Signal Cartography — Architecture Canvas  v2  (refined)
SC_ELAN_LSKA_TSCG  x  DetectCAI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
import numpy as np

BG       = "#0A0A0E"
LINE     = "#D4C89A"
DIM_LINE = "#2E2B24"
ACCENT1  = "#C8974A"
ACCENT2  = "#4A7FA8"
ACCENT3  = "#7A5EA8"
NODE_BG  = "#111119"
SPLIT_C  = "#7AB8D4"
RED_C    = "#B84A4A"
TEXT_DIM = "#5A5447"

UBUNTU_THIN  = "/usr/share/fonts/truetype/ubuntu/Ubuntu-Th.ttf"
UBUNTU_LIGHT = "/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf"
DEJAVU_MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
fp_thin  = FontProperties(fname=UBUNTU_THIN)
fp_light = FontProperties(fname=UBUNTU_LIGHT)
fp_mono  = FontProperties(fname=DEJAVU_MONO)

W, H, DPI = 32, 22, 180
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG); ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect("equal"); ax.axis("off")

def hline(x0,x1,y,c=DIM_LINE,lw=0.4,a=1.0,z=1):
    ax.plot([x0,x1],[y,y],color=c,lw=lw,alpha=a,zorder=z,solid_capstyle="round")
def vline(x,y0,y1,c=DIM_LINE,lw=0.4,a=1.0,z=1):
    ax.plot([x,x],[y0,y1],color=c,lw=lw,alpha=a,zorder=z,solid_capstyle="round")
def arrow(x0,y0,x1,y1,c=LINE,lw=0.8,h=0.12,z=6):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
        arrowprops=dict(arrowstyle=f"->,head_width={h},head_length={h*0.8}",
                        color=c,lw=lw,connectionstyle="arc3,rad=0"),zorder=z)
def poly(pts,c=LINE,lw=0.7,z=3):
    xs,ys=[p[0] for p in pts],[p[1] for p in pts]
    ax.plot(xs,ys,color=c,lw=lw,zorder=z,solid_capstyle="round",solid_joinstyle="round")
def dot(x,y,r=0.07,c=LINE,z=7):
    ax.add_patch(plt.Circle((x,y),r,color=c,zorder=z))
def box(cx,cy,w,h,lb,sb="",fill=NODE_BG,edge=LINE,lw=0.8,fs=9.5,fc=LINE,acc=None,z=4):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle="round,pad=0.02",facecolor=fill,edgecolor=edge,linewidth=lw,zorder=z))
    if acc:
        ax.add_patch(plt.Rectangle((cx-w/2,cy+h/2-0.09),w,0.09,
            facecolor=acc,edgecolor="none",zorder=z+1))
    yo=0.14 if sb else 0.0
    ax.text(cx,cy+yo,lb,ha="center",va="center",fontsize=fs,color=fc,fontproperties=fp_mono,zorder=z+2)
    if sb:
        ax.text(cx,cy-0.26,sb,ha="center",va="center",fontsize=7.2,color=TEXT_DIM,fontproperties=fp_mono,zorder=z+2)
def lbl(x,y,t,s=7.5,c=TEXT_DIM,ha="left",va="center",fp=None,z=8):
    ax.text(x,y,t,ha=ha,va=va,fontsize=s,color=c,fontproperties=fp or fp_mono,zorder=z)

# Background grid
for y in np.arange(0,H+0.5,0.5): hline(0,W,y,lw=0.18,a=0.35)
for x in np.arange(0,W+0.5,0.5): vline(x,0,H,lw=0.18,a=0.35)

# ── HEADER ────────────────────────────────────────────────────────────────────
hline(0.7,W-0.7,H-0.95,c=LINE,lw=0.6,a=0.7)
hline(0.7,W-0.7,H-1.08,c=LINE,lw=0.2,a=0.3)
ax.text(4.5,H-0.60,"SIGNAL CARTOGRAPHY",ha="center",va="center",fontsize=22,color=LINE,fontproperties=fp_thin,zorder=10)
vline(9.5,H-0.95,H-0.25,c=LINE,lw=0.3,a=0.4)
ax.text(10.0,H-0.58,"SC-ELAN-LSKA-TSCG  x  DetectCAI",ha="left",va="center",fontsize=10,color=TEXT_DIM,fontproperties=fp_mono,zorder=10)
lbl(W-0.8,H-0.58,"fig. I / 2026",s=8,c=TEXT_DIM,ha="right",z=10)
lbl(1.0,H-1.30,"I.   SC_ELAN_LSKA_TSCG",s=9,c=ACCENT1,fp=fp_light,z=10)
lbl(W/2+0.8,H-1.30,"II.  DetectCAI  (Class-Adaptive Interaction)",s=9,c=ACCENT2,fp=fp_light,z=10)
vline(W/2,0.5,H-1.1,lw=0.5,a=0.5)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL I — SC_ELAN_LSKA_TSCG
# ══════════════════════════════════════════════════════════════════════════════
LX=7.2; TOP=H-2.0; BW=3.8; BH=0.72; GAP=1.22

# Input
Y0=TOP
box(LX,Y0,BW,BH,"x  [ B x C x H x W ]","input feature tensor",fill="#14142A",edge=LINE,fs=9.5)

# cv1
Y1=Y0-GAP
arrow(LX,Y0-BH/2,LX,Y1+BH/2,c=ACCENT1,lw=0.9)
lbl(LX+BW/2+0.12,(Y0+Y1)/2,"C -> C",s=7,c=ACCENT1)
box(LX,Y1,BW,BH,"cv1   Conv 1x1","projection",edge=ACCENT1,acc=ACCENT1)

# SPLIT
Y2=Y1-GAP*0.7
arrow(LX,Y1-BH/2,LX,Y2+BH/2,c=LINE,lw=0.6)
box(LX,Y2,BW,0.44,"SPLIT  /  2","chunk(dim=1) -> [y0 | y1]",fill="#0C0C1A",edge=LINE)

LBX=LX-2.35; Y_SP=Y2-0.22-0.18
dot(LX-BW/2*0.55,Y2-0.22,c=SPLIT_C)
poly([(LX-BW/2*0.55,Y2-0.22),(LBX,Y2-0.22),(LBX,Y_SP-0.05)],c=SPLIT_C,lw=0.75)
lbl(LBX+0.11,Y2-0.22+0.18,"y0  (1/2 C)",s=7,c=SPLIT_C)
arrow(LX,Y2-BH/2*0.8,LX,Y_SP,c=LINE,lw=0.6)
lbl(LX+0.11,Y_SP+0.14,"y1",s=8,c=LINE)

# cv2
Y3=Y_SP-GAP*0.82
arrow(LX,Y_SP,LX,Y3+BH/2,c=LINE,lw=0.6)
box(LX,Y3,BW,BH,"cv2   ContextAwareRepConv","PolarZone multi-branch  1/2C -> 1/2C",edge=ACCENT1,acc=ACCENT1,fs=8.5)
Y2TAP=Y3-BH/2-0.08
dot(LX,Y2TAP,c=LINE)
lbl(LX+0.13,Y2TAP+0.14,"y2",s=8,c=LINE)

# cv3
Y4=Y3-GAP*0.82
arrow(LX,Y3-BH/2,LX,Y4+BH/2,c=LINE,lw=0.6)
box(LX,Y4,BW,BH,"cv3   ContextAwareRepConv","PolarZone multi-branch  1/2C -> 1/2C",edge=ACCENT1,acc=ACCENT1,fs=8.5)
Y3TAP=Y4-BH/2-0.08
dot(LX,Y3TAP,c=LINE)
lbl(LX+0.13,Y3TAP+0.14,"y3",s=8,c=LINE)

# CONCAT
Y_CAT=Y4-GAP*0.72
SPUR_Y2=LX+0.8; SPUR_Y3=LX+1.5
# y1 spur (just continue down from Y_SP)
poly([(LX,Y_SP),(LX,Y_CAT+0.22)],c=LINE,lw=0.4)
# y2
dot(LX,Y2TAP,c=LINE)
poly([(LX,Y2TAP),(SPUR_Y2,Y2TAP),(SPUR_Y2,Y_CAT+0.22)],c=LINE,lw=0.5)
# y3
dot(LX,Y3TAP,c=LINE)
poly([(LX,Y3TAP),(SPUR_Y3,Y3TAP),(SPUR_Y3,Y_CAT+0.22)],c=LINE,lw=0.5)
# y0 left
poly([(LBX,Y_SP-0.05),(LBX,Y_CAT+0.22)],c=SPLIT_C,lw=0.75)
# horizontal merge bar
hline(LBX,SPUR_Y3,Y_CAT+0.22,c=LINE,lw=0.55)
dot(LBX,Y_CAT+0.22,c=SPLIT_C)
arrow(LX+0.3,Y_CAT+0.24,LX+0.3,Y_CAT+BH/2,c=LINE,lw=0.5,h=0.08)
box(LX,Y_CAT,BW,BH,"CONCAT   [ y0 || y1 || y2 || y3 ]","torch.cat(dim=1)  ->  2C",fill="#0C0C18",edge=LINE)

# cv4
Y5=Y_CAT-GAP*0.78
arrow(LX,Y_CAT-BH/2,LX,Y5+BH/2,c=ACCENT1,lw=0.9)
lbl(LX+BW/2+0.12,(Y_CAT+Y5)/2,"2C -> C",s=7,c=ACCENT1)
box(LX,Y5,BW,BH,"cv4   Conv 1x1","aggregate  ->  feat",edge=ACCENT1,acc=ACCENT1)

# feat dot + bypass
Y_FEAT=Y5-BH/2-0.08
dot(LX,Y_FEAT,c=ACCENT2,r=0.10)
lbl(LX+0.17,Y_FEAT+0.013,"feat",s=9,c=ACCENT2)
BPX=LX-BW/2-0.55
poly([(LX-BW/2,Y_FEAT),(BPX,Y_FEAT)],c=RED_C,lw=0.7)

# LSKA
Y6=Y5-GAP*1.12
arrow(LX,Y_FEAT,LX,Y6+BH/2*1.15,c=ACCENT2,lw=0.9)
box(LX,Y6,BW,BH*1.15,"LSKA  attention","Large Separable Kernel Attention",fill="#08101C",edge=ACCENT2,acc=ACCENT2,fc=ACCENT2,fs=10)

# LSKA sub-diagram
LK_CX=LX+BW/2+1.65; LK_W=2.1; LK_H=0.36; LK_G=0.54; LK_Y0=Y6+1.10
hline(LX+BW/2,LK_CX-LK_W/2,Y6,c=DIM_LINE,lw=0.35,a=0.6)
lbl(LK_CX,LK_Y0+0.42,"[ LSKA internals ]",s=7.5,c=ACCENT2,ha="center")
lska_rows=[("clone  ->  u","identity branch"),("conv0h  1x(k//2+1)","horiz local"),
    ("conv0v  (k//2+1)x1","vert local"),("conv_sp_h  dil","spatial H dilated"),
    ("conv_sp_v  dil","spatial V dilated"),("conv1  1x1","channel proj"),
    ("u  *  attn","multiplicative gate")]
for i,(la,su) in enumerate(lska_rows):
    ky=LK_Y0-i*LK_G
    ec=ACCENT2 if (i==0 or i==len(lska_rows)-1) else DIM_LINE
    fc2=ACCENT2 if i==len(lska_rows)-1 else LINE
    box(LK_CX,ky,LK_W,LK_H,la,su,edge=ec,fc=fc2,fs=7)
    if i<len(lska_rows)-1:
        arrow(LK_CX,ky-LK_H/2,LK_CX,ky-LK_G+LK_H/2,c=ACCENT2,lw=0.45,h=0.06)

Y_CTX=Y6-BH/2*1.15-0.12
dot(LX,Y_CTX,c=ACCENT2,r=0.10)
lbl(LX+0.17,Y_CTX+0.013,"context",s=9,c=ACCENT2)

Y_BPB=Y_CTX-GAP*0.55
poly([(BPX,Y_FEAT),(BPX,Y_BPB)],c=RED_C,lw=0.7)
lbl(BPX-0.08,(Y_FEAT+Y_BPB)/2,"feat\nresidual",s=7,c=RED_C,ha="right")

# TSCG
Y7=Y_CTX-GAP*0.82
arrow(LX,Y_CTX,LX,Y7+BH/2*1.15,c=ACCENT2,lw=0.8)
box(LX,Y7,BW,BH*1.15,"TSCG","TinySelectiveContextGate",fill="#120A24",edge=ACCENT3,acc=ACCENT3,fc=ACCENT3,fs=11)
lbl(LX,Y7-0.30,"gate = sigmoid( DWConv->Conv1x1->BN->SiLU->Conv1x1 )",s=6.5,c=TEXT_DIM,ha="center")
poly([(BPX,Y_BPB),(BPX,Y7)],c=RED_C,lw=0.7)
arrow(BPX,Y7,LX-BW/2,Y7,c=RED_C,lw=0.7,h=0.09)
lbl(BPX-0.08,Y7+0.14,"feat",s=7,c=RED_C,ha="right")
lbl(LX,Y7-0.80,"output  =  feat  +  gate * (context - feat)",s=8,c=ACCENT3,ha="center")

# Output
Y8=Y7-GAP*0.98
arrow(LX,Y7-BH/2*1.15,LX,Y8+BH/2,c=ACCENT3,lw=0.9,h=0.13)
box(LX,Y8,BW,BH,"output   [ B x C x H x W ]","context-enhanced features",fill="#1A0A28",edge=ACCENT3)

# ContextAwareRepConv explainer
CA_CX=2.4; CA_Y0=Y3+0.5; CA_W=1.7; CA_H=0.38; CA_G=0.55
lbl(CA_CX,CA_Y0+0.55,"ContextAwareRepConv",s=8,c=ACCENT1,ha="center")
lbl(CA_CX,CA_Y0+0.32,"(training branches)",s=6.5,c=TEXT_DIM,ha="center")
ca_rows=[("3x3 dense","rbr_dense"),("3x3 dilation=2","rbr_dilated ctx"),("Sum  BN  SiLU","fused activation")]
for i,(la,su) in enumerate(ca_rows):
    cy_=CA_Y0-i*CA_G
    box(CA_CX,cy_,CA_W,CA_H,la,su,edge=ACCENT1 if i==2 else DIM_LINE,fs=7.5,acc=ACCENT1 if i==2 else None)
    if i<len(ca_rows)-1:
        arrow(CA_CX,cy_-CA_H/2,CA_CX,cy_-CA_G+CA_H/2,c=ACCENT1,lw=0.4,h=0.06)
hline(CA_CX+CA_W/2,LX-BW/2,Y3,c=DIM_LINE,lw=0.35,a=0.6)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL II — DetectCAI
# ══════════════════════════════════════════════════════════════════════════════
RX=W/2+8.4; TOP2=TOP; BW2=4.0; BH2=0.72; GAP2=1.15

SCALE_CFG=[("P3","160 x 160",RX-3.6,SPLIT_C),("P4","80 x  80",RX,ACCENT2),("P5","40 x  40",RX+3.6,LINE)]
for name,res,sx,col in SCALE_CFG:
    box(sx,TOP2,2.5,BH2,f"{name}  [ {res} ]","neck feature map",fill="#08141C",edge=col,fc=col,fs=8.5)

Y_PE=TOP2-GAP2*0.82
for name,res,sx,col in SCALE_CFG:
    arrow(sx,TOP2-BH2/2,sx,Y_PE+BH2/2,c=col,lw=0.55,h=0.08)
    box(sx,Y_PE,2.5,BH2,"AvgPool2d  +  Linear","C -> nc  (class logits)",edge=ACCENT2,fc=ACCENT2,fs=8)

Y_SF=Y_PE-GAP2*0.72
for name,res,sx,col in SCALE_CFG:
    poly([(sx,Y_PE-BH2/2),(sx,Y_SF+BH2/2*0.5)],c=ACCENT2,lw=0.45)
hline(SCALE_CFG[0][2],SCALE_CFG[-1][2],Y_SF+BH2/2*0.5,c=ACCENT2,lw=0.45)
arrow(RX,Y_SF+BH2/2*0.5,RX,Y_SF+BH2/2,c=ACCENT2,lw=0.65,h=0.09)
box(RX,Y_SF,BW2,BH2,"mean  ->  Softmax","-> pred_prior  [ nc ]",fill="#08101C",edge=ACCENT2,acc=ACCENT2,fc=ACCENT2)
lbl(RX+BW2/2+0.15,Y_SF+0.05,"EMA: p <- m*p + (1-m)*p_hat",s=7,c=TEXT_DIM)

Y_PR=Y_SF-GAP2*0.72
arrow(RX,Y_SF-BH2/2,RX,Y_PR+BH2/2,c=ACCENT2,lw=0.65)
box(RX,Y_PR,BW2,BH2,"cai_class_prior   p   [ nc ]","EMA-updated class distribution",edge=ACCENT2,fc=ACCENT2,fs=8.5)

Y_EM=Y_PR-GAP2*0.76
arrow(RX,Y_PR-BH2/2,RX,Y_EM+BH2/2,c=ACCENT3,lw=0.65)
box(RX,Y_EM,BW2,BH2,"p @ Embedding(nc, d)  ->  class_ctx","class-weighted embedding  [ embed_dim ]",fill="#130B20",edge=ACCENT3,acc=ACCENT3,fc=ACCENT3,fs=8)
lbl(RX+BW2/2+0.15,Y_EM+0.07,"tail_weight = sum( p[i] * tail_mask[i] )",s=7,c=TEXT_DIM)

Y_GT=Y_EM-GAP2*1.08
lbl(RX,Y_GT+0.52,"- - -  per-scale gate computation  - - -",s=7.5,c=TEXT_DIM,ha="center")
GSX=[c[2] for c in SCALE_CFG]; GC=[c[3] for c in SCALE_CFG]

for i,(sx,col) in enumerate(zip(GSX,GC)):
    poly([(SCALE_CFG[i][2],TOP2-BH2/2),(sx,Y_GT+BH2/2)],c=col,lw=0.4)
    box(sx,Y_GT,2.4,BH2,"AvgPool2d  ->  pooled",f"scale {i} spatial collapse",edge=col,fc=col,fs=8)
    Y_BG=Y_GT-GAP2*0.82
    arrow(sx,Y_GT-BH2/2,sx,Y_BG+BH2/2,c=ACCENT1,lw=0.55,h=0.07)
    box(sx,Y_BG,2.4,BH2,"base_gate","Conv->SiLU->Conv  sigmoid",edge=ACCENT1,acc=ACCENT1,fs=8)
    Y_CG=Y_BG-GAP2*0.82
    poly([(RX,Y_EM-BH2/2),(sx,Y_CG+BH2/2)],c=ACCENT3,lw=0.35)
    box(sx,Y_CG,2.4,BH2,"cond_gate","[pooled||ctx]->Lin->SiLU->Lin",fill="#130B20",edge=ACCENT3,acc=ACCENT3,fs=7.5)
    Y_SUM=Y_CG-GAP2*0.80
    arrow(sx,Y_BG-BH2/2,sx,Y_SUM+BH2,c=ACCENT1,lw=0.5,h=0.07)
    arrow(sx,Y_CG-BH2/2,sx,Y_SUM+BH2/2,c=ACCENT3,lw=0.5,h=0.07)
    box(sx,Y_SUM,2.4,BH2,"gate = 1 + a*G_b + b*G_c*tw","a=0.15  b=0.30  tw=tail_weight",fill="#0C0C0C",edge=LINE,fs=7)
    Y_MX=Y_SUM-GAP2*0.78
    arrow(sx,Y_SUM-BH2/2,sx,Y_MX+BH2/2,c=LINE,lw=0.65,h=0.09)
    box(sx,Y_MX,2.4,BH2,"xi  *  gate","element-wise reweighting",edge=col,fs=8)

Y_MRG=Y_MX-GAP2*0.76
for sx in GSX:
    arrow(sx,Y_MX-BH2/2,sx,Y_MRG+BH2/2*0.5,c=LINE,lw=0.45,h=0.07)
hline(GSX[0],GSX[-1],Y_MRG+BH2/2*0.5,c=LINE,lw=0.45)
arrow(RX,Y_MRG+BH2/2*0.5,RX,Y_MRG+BH2/2,c=LINE,lw=0.65,h=0.09)
box(RX,Y_MRG,BW2,BH2,"Detect.forward( x_cai )","standard YOLO detection head",fill="#071407",edge=ACCENT1,acc=ACCENT1,fc=ACCENT1,fs=9)

BKX0=RX-BW2/2-0.3; BKX1=RX+BW2/2+0.3; BKY0=Y_EM+BH2/2; BKY1=Y_MRG-BH2/2-0.05
vline(BKX0,BKY1,BKY0,lw=0.4,a=0.55)
vline(BKX1,BKY1,BKY0,lw=0.4,a=0.55)
hline(BKX0,BKX0+0.22,BKY0,lw=0.4,a=0.55)
hline(BKX0,BKX0+0.22,BKY1,lw=0.4,a=0.55)
hline(BKX1-0.22,BKX1,BKY0,lw=0.4,a=0.55)
hline(BKX1-0.22,BKX1,BKY1,lw=0.4,a=0.55)
lbl(BKX0-0.1,(BKY0+BKY1)/2,"training\nonly",s=7,c=TEXT_DIM,ha="right")

Y_OUT=Y_MRG-GAP2*0.76
arrow(RX,Y_MRG-BH2/2,RX,Y_OUT+BH2/2,c=ACCENT1,lw=0.9,h=0.13)
box(RX,Y_OUT,BW2,BH2,"predictions   [ cls  reg  dfl ]","boxes + class scores",fill="#0A1A09",edge=ACCENT1,fs=9)

# ── LEGEND + FOOTER ───────────────────────────────────────────────────────────
# Place legend just below the lowest content element
CONTENT_BOT = min(Y8 - BH/2, Y_OUT - BH2/2)
LY = CONTENT_BOT - 0.85
hline(0.7,W-0.7,LY+0.38,c=LINE,lw=0.3,a=0.4)
leg_defs=[(ACCENT1,"projection / reparam path"),(ACCENT2,"attention / prior path"),
    (ACCENT3,"gating / adaptation"),(SPLIT_C,"split channels"),(RED_C,"residual bypass")]
lx=2.0
for col,txt in leg_defs:
    ax.plot([lx,lx+0.65],[LY,LY],color=col,lw=2.5,solid_capstyle="round",zorder=9)
    lbl(lx+0.80,LY,txt,s=7.5,c=TEXT_DIM,ha="left")
    lx+=4.8
lbl(W-0.8,LY,"Signal Cartography  |  SC-ELAN-LSKA-TSCG x DetectCAI  |  ultralytics-SC",s=6.5,c=TEXT_DIM,ha="right")

# Corner ticks (relative to legend bottom)
CNR_BOT = LY - 0.35
for xx,yy in [(0.45,CNR_BOT),(0.45,H-0.45),(W-0.45,CNR_BOT),(W-0.45,H-0.45)]:
    ax.plot([xx-0.28,xx+0.28],[yy,yy],color=LINE,lw=0.5,alpha=0.5)
    ax.plot([xx,xx],[yy-0.28,yy+0.28],color=LINE,lw=0.5,alpha=0.5)

# Dot-grid accent
for dx in np.arange(0.9,W-0.5,1.5):
    for dy in np.arange(LY-0.2,H-1.3,1.5):
        ax.plot(dx,dy,".",color=DIM_LINE,ms=0.7,alpha=0.6,zorder=0)

# Trim canvas: set ylim tightly around content
ax.set_ylim(LY - 0.45, H)

out="/mnt/nas/Programming/ultralytics-SC/canvas/architecture_canvas.png"
fig.savefig(out,dpi=DPI,bbox_inches="tight",facecolor=BG,edgecolor="none")
print(f"Saved: {out}")
plt.close(fig)
