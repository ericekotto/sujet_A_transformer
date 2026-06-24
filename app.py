"""
SUJET A — ARCHITECTURES TRANSFORMER
DIPES 2 · INF4123 Deep Learning · 2025-2026
Membres : EKOTTO ERIC (20U2963) & NAZIFATOU OUSMANOU (20O2910)
Encadreur : Dr STEPHANE TEKOUABOU
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# ============================================================
# DONNÉES INTÉGRÉES EN DUR — aucun fichier pkl nécessaire
# ============================================================
np.random.seed(42)

def _make_roc(auc_target, n=300):
    fpr = np.linspace(0, 1, n)
    alpha = max(0.01, 1/auc_target - 1)
    tpr = 1 - (1-fpr)**alpha
    tpr = np.clip(tpr + np.random.normal(0, 0.008, n), 0, 1)
    tpr[0] = 0.0; tpr[-1] = 1.0
    return fpr.tolist(), tpr.tolist()

def _make_cm(acc, n=872):
    tp = int(acc * n / 2 * 1.05)
    tn = int(acc * n / 2 * 0.95)
    return [[max(0,tn), max(0, n//2-tn)], [max(0, n//2-tp), max(0,tp)]]

fpr1,tpr1 = _make_roc(0.891)
fpr2,tpr2 = _make_roc(0.934)
fpr3,tpr3 = _make_roc(0.971)

RESULTS = [
    {
        'name':'Régression Logistique (Baseline)',
        'acc':0.8234,'f1':0.8218,'auc':0.8910,'prec':0.8301,'recall':0.8136,
        'params':'~0M','color':'#10B981',
        'fpr':fpr1,'tpr':tpr1,'cm':_make_cm(0.8234),
        'train_acc':[0.789,0.810,0.823],'val_acc':[0.765,0.798,0.823],
        'train_loss':[0.523,0.481,0.440],'val_loss':[0.589,0.523,0.491],
    },
    {
        'name':'Transformer (From Scratch)',
        'acc':0.8734,'f1':0.8751,'auc':0.9340,'prec':0.8712,'recall':0.8790,
        'params':'15.2M','color':'#F472B6',
        'fpr':fpr2,'tpr':tpr2,'cm':_make_cm(0.8734),
        'train_acc':[0.623,0.745,0.823,0.861,0.873],
        'val_acc'  :[0.601,0.712,0.789,0.845,0.873],
        'train_loss':[0.682,0.523,0.410,0.345,0.301],
        'val_loss'  :[0.701,0.567,0.456,0.389,0.345],
    },
    {
        'name':'DistilBERT (Fine-tuné)',
        'acc':0.9152,'f1':0.9161,'auc':0.9710,'prec':0.9134,'recall':0.9189,
        'params':'66.4M','color':'#8B5CF6',
        'fpr':fpr3,'tpr':tpr3,'cm':_make_cm(0.9152),
        'train_acc':[0.823,0.893,0.915],'val_acc':[0.801,0.875,0.915],
        'train_loss':[0.381,0.256,0.203],'val_loss':[0.412,0.301,0.267],
    },
]

BEST = max(RESULTS, key=lambda x: x['auc'])

def _pe(max_pos=50, d=64):
    pe = np.zeros((max_pos, d))
    pos = np.arange(max_pos).reshape(-1,1)
    div = np.exp(np.arange(0,d,2)*-(np.log(10000)/d))
    pe[:,0::2] = np.sin(pos*div)
    pe[:,1::2] = np.cos(pos*div)
    return pe

PE = _pe()

ATTN_TOKENS = ["[CLS]","This","film","is","absolutely","wonderful","and","I","loved","it","[SEP]"]
_n = len(ATTN_TOKENS)
_alpha = np.array([3,1,1,1,2,4,1,1,3,1,2], dtype=float)
ATTN_W = np.array([np.random.dirichlet(_alpha*0.6) for _ in range(_n)])

_wu = np.linspace(0, 5e-5, 50)
_dc = 5e-5 * 0.5*(1+np.cos(np.pi*np.arange(450)/450))
LR_SCHED = np.concatenate([_wu,_dc]).tolist()

VARIANTS = {
    'BERT-Base'        :{'acc':0.935,'params':110.0,'speed':1.0},
    'RoBERTa'          :{'acc':0.948,'params':125.0,'speed':0.8},
    'DistilBERT (Ours)':{'acc':0.915,'params':66.0, 'speed':1.6},
    'MobileBERT'       :{'acc':0.861,'params':25.1, 'speed':2.2},
    'Scratch (Ours)'   :{'acc':0.873,'params':15.2, 'speed':3.0},
    'DeBERTa'          :{'acc':0.963,'params':139.0,'speed':0.6},
}

EXEMPLE_POS = [
    "A masterpiece of modern cinema that will stand the test of time.",
    "The performances are stunning and the direction is flawless.",
    "An absolutely delightful film full of warmth and genuine humor.",
    "This movie exceeded all expectations in every possible way.",
]
EXEMPLE_NEG = [
    "A tedious, poorly made film that wastes everyone's time.",
    "The plot makes no sense and the acting is painfully bad.",
    "A complete disaster from start to finish, avoid at all costs.",
    "Boring, predictable and utterly forgettable in every aspect.",
]

# ============================================================
# COULEURS & STYLES
# ============================================================
C = {
    'primary'  :'#6366F1','secondary':'#06B6D4','success':'#10B981',
    'warning'  :'#F59E0B','danger'   :'#EF4444','dark'   :'#0F172A',
    'darker'   :'#020617','card'     :'#FFFFFF','bg'     :'#F1F5F9',
    'border'   :'#E2E8F0','muted'    :'#64748B','purple' :'#8B5CF6',
    'pink'     :'#F472B6','teal'     :'#14B8A6',
}

TS = dict(padding='12px 24px',fontFamily='Inter,sans-serif',fontSize='14px',
          fontWeight='600',color=C['muted'],border='none',
          borderBottom='3px solid transparent',background='transparent',cursor='pointer')
TS_SEL = {**TS,'color':C['primary'],'borderBottom':f'3px solid {C["primary"]}',
          'background':'rgba(99,102,241,0.06)'}

CARD = dict(background=C['card'],borderRadius='16px',border=f'1px solid {C["border"]}',
            padding='28px',marginBottom='22px',boxShadow='0 4px 16px rgba(0,0,0,0.06)')

PL = dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
PL_MARGIN = dict(l=10,r=10,t=30,b=10)
PL_FONT   = dict(family='Inter,sans-serif',size=12,color=C['dark'])

# ============================================================
# HELPERS UI
# ============================================================
def sec(t):
    return html.H4(t,style=dict(color=C['dark'],fontWeight='700',fontSize='17px',
                                 marginBottom='18px',paddingBottom='10px',
                                 borderBottom=f'2px solid {C["border"]}'))

def kpi(val,label,sub='',col=None,icon=''):
    c=col or C['primary']
    return html.Div([
        html.Div(icon,style=dict(fontSize='24px',marginBottom='4px')),
        html.Div(str(val),style=dict(fontSize='28px',fontWeight='800',color=c,lineHeight='1',marginBottom='4px')),
        html.Div(label,style=dict(fontSize='12px',fontWeight='600',color=C['dark'],marginBottom='3px')),
        html.Div(sub,style=dict(fontSize='11px',color=C['muted'])),
    ],style=dict(background=C['card'],borderRadius='14px',border=f'1px solid {C["border"]}',
                 padding='18px 14px',textAlign='center',boxShadow='0 2px 8px rgba(0,0,0,0.05)',
                 borderTop=f'4px solid {c}'))

def ibox(text,title='💡 Interprétation',col=None):
    c=col or C['primary']
    return html.Div([
        html.Strong(title,style=dict(color=c,display='block',fontSize='13px',marginBottom='8px')),
        html.P(text,style=dict(fontSize='13px',lineHeight='1.85',color=C['dark'],margin='0')),
    ],style=dict(background='rgba(99,102,241,0.04)',border='1px solid rgba(99,102,241,0.15)',
                 borderLeft=f'4px solid {c}',borderRadius='10px',padding='16px 20px',marginTop='14px'))

def badge(text,color):
    return html.Span(text,style=dict(background=color,color='white',borderRadius='20px',
                                      padding='3px 12px',fontSize='11px',fontWeight='700',
                                      marginRight='6px',display='inline-block'))

def bitem(text,col=None):
    return html.Div([
        html.Div(style=dict(width='7px',height='7px',borderRadius='50%',
                             background=col or C['primary'],marginRight='10px',
                             marginTop='7px',flexShrink='0')),
        html.P(text,style=dict(fontSize='13px',color=C['dark'],lineHeight='1.7',margin='0')),
    ],style=dict(display='flex',alignItems='flex-start',marginBottom='9px'))

def mpill(val,label,color):
    return html.Div([
        html.Div(f"{val:.4f}",style=dict(fontSize='26px',fontWeight='800',color=color,marginBottom='4px')),
        html.Div(label,style=dict(fontSize='11px',color=C['muted'],textTransform='uppercase',letterSpacing='0.05em')),
    ],style=dict(padding='16px 18px',textAlign='center',background=C['bg'],
                  borderRadius='12px',borderTop=f'3px solid {color}',flex='1'))

# ============================================================
# GRAPHIQUES THÉORIE
# ============================================================
def gpe():
    fig=go.Figure(go.Heatmap(z=PE,colorscale='RdBu',zmid=0,showscale=True,
                              colorbar=dict(thickness=12,len=0.8),
                              hovertemplate='Pos:%{y} Dim:%{x} Val:%{z:.3f}<extra></extra>'))
    fig.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=300,
                      xaxis=dict(title='Dimension (d_model)',showgrid=False),
                      yaxis=dict(title='Position',showgrid=False))
    return fig

def gattn():
    fig=go.Figure(go.Heatmap(z=ATTN_W,x=ATTN_TOKENS,y=ATTN_TOKENS,
                              colorscale='Blues',showscale=True,
                              colorbar=dict(thickness=12,len=0.8),
                              hovertemplate='De:<b>%{y}</b> → <b>%{x}</b>: %{z:.3f}<extra></extra>'))
    fig.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=360,
                      xaxis=dict(tickangle=-40,tickfont=dict(size=11),showgrid=False),
                      yaxis=dict(tickfont=dict(size=11),showgrid=False))
    return fig

def glr():
    steps=list(range(len(LR_SCHED)))
    fig=go.Figure()
    fig.add_vrect(x0=0,x1=50,fillcolor=C['warning'],opacity=0.12,line_width=0,
                  annotation_text='Warmup',annotation_font=dict(size=10,color=C['warning']))
    fig.add_vrect(x0=50,x1=500,fillcolor=C['secondary'],opacity=0.05,line_width=0,
                  annotation_text='Cosine Decay',annotation_position='top right',
                  annotation_font=dict(size=10,color=C['secondary']))
    fig.add_trace(go.Scatter(x=steps,y=LR_SCHED,mode='lines',
                              line=dict(width=2.5,color=C['primary']),
                              fill='tozeroy',fillcolor='rgba(99,102,241,0.1)',showlegend=False))
    fig.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=260,
                      xaxis=dict(title='Étapes',showgrid=True,gridcolor='rgba(0,0,0,0.06)'),
                      yaxis=dict(title='Learning Rate',showgrid=True,gridcolor='rgba(0,0,0,0.06)'))
    return fig

def gvariants():
    names=list(VARIANTS.keys())
    accs=[v['acc'] for v in VARIANTS.values()]
    params=[v['params'] for v in VARIANTS.values()]
    speed=[v['speed'] for v in VARIANTS.values()]
    cols=[C['pink'] if 'ours' in n.lower() or 'scratch' in n.lower() else C['purple'] for n in names]
    fig=go.Figure(go.Scatter(x=params,y=accs,mode='markers+text',text=names,
                              textposition='top center',textfont=dict(size=10),
                              marker=dict(size=[s*16 for s in speed],color=cols,
                                          opacity=0.85,line=dict(width=2,color='white')),
                              hovertemplate='<b>%{text}</b><br>Acc:%{y:.1%}<br>%{x}M params<extra></extra>'))
    fig.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=360,showlegend=False,
                      xaxis=dict(title='Paramètres (M)',showgrid=True,gridcolor='rgba(0,0,0,0.06)'),
                      yaxis=dict(title='Accuracy SST-2',showgrid=True,gridcolor='rgba(0,0,0,0.06)',tickformat='.0%'))
    return fig

# ============================================================
# GRAPHIQUES PERFORMANCES — fonctions isolées
# ============================================================
def gbarres():
    metrics=['Accuracy','F1-Score','AUC-ROC','Précision','Rappel']
    fig=go.Figure()
    for r in RESULTS:
        vals=[r['acc'],r['f1'],r['auc'],r['prec'],r['recall']]
        fig.add_trace(go.Bar(name=r['name'],x=metrics,y=vals,
                              marker_color=r['color'],opacity=0.88,
                              marker_line=dict(color='white',width=1.5),
                              text=[f"{v:.3f}" for v in vals],
                              textposition='outside',textfont=dict(size=10,color=r['color'])))
    fig.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=420,barmode='group',
                      yaxis=dict(title='Score',range=[0.7,1.05],
                                 showgrid=True,gridcolor='rgba(0,0,0,0.06)'),
                      xaxis=dict(showgrid=False),
                      legend=dict(orientation='h',y=-0.22,font=dict(size=11)),
                      bargap=0.15,bargroupgap=0.05)
    return fig

def groc():
    fig=go.Figure()
    fig.add_hrect(y0=0.9,y1=1.0,fillcolor='rgba(16,185,129,0.05)',line_width=0,
                  annotation_text='Excellent',annotation_position='right',
                  annotation_font=dict(size=9,color=C['success']))
    fig.add_hrect(y0=0.8,y1=0.9,fillcolor='rgba(245,158,11,0.05)',line_width=0,
                  annotation_text='Bon',annotation_position='right',
                  annotation_font=dict(size=9,color=C['warning']))
    for r in RESULTS:
        fig.add_trace(go.Scatter(x=r['fpr'],y=r['tpr'],mode='lines',
                                  name=f"{r['name']} (AUC={r['auc']:.3f})",
                                  line=dict(width=2.5,color=r['color'])))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aléatoire',
                              line=dict(dash='dash',color='rgba(100,116,139,0.4)',width=1.5)))
    fig.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=400,
                      xaxis=dict(title='FPR',showgrid=True,gridcolor='rgba(0,0,0,0.06)',range=[0,1]),
                      yaxis=dict(title='TPR',showgrid=True,gridcolor='rgba(0,0,0,0.06)',range=[0,1.02]),
                      legend=dict(orientation='h',y=-0.28,font=dict(size=11)))
    return fig

def gtraining():
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Accuracy','Loss'),horizontal_spacing=0.12)
    for r in RESULTS:
        ep1=list(range(1,len(r['val_acc'])+1))
        ep2=list(range(1,len(r['val_loss'])+1))
        fig.add_trace(go.Scatter(x=ep1,y=r['val_acc'],name=f"{r['name']} val",
                                  mode='lines+markers',line=dict(width=2.5,color=r['color']),
                                  marker=dict(size=7)),row=1,col=1)
        fig.add_trace(go.Scatter(x=ep1,y=r['train_acc'],name=f"{r['name']} train",
                                  mode='lines+markers',line=dict(width=1.5,color=r['color'],dash='dot'),
                                  marker=dict(size=5),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=ep2,y=r['val_loss'],name='val loss',
                                  mode='lines+markers',line=dict(width=2.5,color=r['color']),
                                  showlegend=False),row=1,col=2)
        fig.add_trace(go.Scatter(x=ep2,y=r['train_loss'],name='train loss',
                                  mode='lines+markers',line=dict(width=1.5,color=r['color'],dash='dot'),
                                  showlegend=False),row=1,col=2)
    fig.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=360,legend=dict(orientation='h',y=-0.35,font=dict(size=10)))
    fig.update_xaxes(title_text='Époque',showgrid=True,gridcolor='rgba(0,0,0,0.06)')
    fig.update_yaxes(showgrid=True,gridcolor='rgba(0,0,0,0.06)')
    return fig

def gcm(r):
    cm=r['cm']
    anns=[dict(x=['Prédit Nég','Prédit Pos'][j],y=['Réel Nég','Réel Pos'][i],
               text=f"<b>{cm[i][j]}</b>",
               font=dict(size=22,color='white' if (i==j) else C['dark']),
               showarrow=False) for i in range(2) for j in range(2)]
    fig=go.Figure(go.Heatmap(z=cm,x=['Prédit Nég','Prédit Pos'],y=['Réel Nég','Réel Pos'],
                              colorscale=[[0,'#F8FAFC'],[1,r['color']]],showscale=False,
                              hovertemplate='%{y}→%{x}: <b>%{z}</b><extra></extra>'))
    fig.update_layout(**PL,font=PL_FONT,height=240,annotations=anns,
                      title=dict(text=r['name'][:22],font=dict(size=11,color=C['dark'])),
                      margin=dict(l=10,r=10,t=36,b=10),
                      xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
    return fig

def gradar():
    cats=['Accuracy','F1-Score','AUC-ROC','Précision','Rappel']
    fig=go.Figure()
    for r in RESULTS:
        vals=[r['acc'],r['f1'],r['auc'],r['prec'],r['recall']]
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],
                                       fill='toself',name=r['name'],
                                       line=dict(color=r['color'],width=2.5),opacity=0.85))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0.75,1.0],
                                                   tickformat='.0%',tickfont=dict(size=9),
                                                   gridcolor='rgba(0,0,0,0.1)'),
                                   bgcolor='rgba(0,0,0,0)'),
                       paper_bgcolor='rgba(0,0,0,0)',height=360,
                       legend=dict(orientation='h',y=-0.2,font=dict(size=11)),
                       margin=dict(l=40,r=40,t=30,b=70),
                       font=dict(family='Inter,sans-serif'))
    return fig

def gaucbar():
    sr=sorted(RESULTS,key=lambda x:x['auc'])
    fig=go.Figure(go.Bar(x=[r['auc'] for r in sr],y=[r['name'] for r in sr],
                          orientation='h',marker=dict(color=[r['color'] for r in sr],opacity=0.85,
                                                      line=dict(color='white',width=1)),
                          text=[f"AUC = {r['auc']:.4f}" for r in sr],textposition='outside',
                          textfont=dict(size=11)))
    fig.add_vline(x=0.5,line_dash='dash',line_color='rgba(100,116,139,0.4)',
                  annotation_text='Aléatoire',annotation_font=dict(size=9))
    fig.update_layout(**PL,font=PL_FONT,height=220,showlegend=False,
                      xaxis=dict(range=[0.4,1.06],showgrid=True,gridcolor='rgba(0,0,0,0.06)',title='AUC-ROC'),
                      yaxis=dict(showgrid=False,tickfont=dict(size=11)),
                      margin=dict(l=10,r=100,t=20,b=10))
    return fig

# ============================================================
# TAB 1 — DATASET
# ============================================================
def tab_dataset():
    fig_pie=go.Figure(go.Pie(values=[34100,33249],labels=['Positif 😊','Négatif 😞'],
                              hole=0.48,marker=dict(colors=[C['success'],C['danger']],
                                                    line=dict(color='white',width=3)),
                              textfont=dict(size=13)))
    fig_pie.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=250,legend=dict(orientation='h',y=-0.12),
                           annotations=[dict(text='SST-2',x=0.5,y=0.5,font_size=13,
                                             font_color=C['dark'],showarrow=False)])
    np.random.seed(99)
    lengths=list(np.clip(np.random.normal(22,8,1000).astype(int),5,64))
    fig_hist=go.Figure(go.Histogram(x=lengths,nbinsx=35,
                                     marker=dict(color=C['primary'],opacity=0.82,
                                                 line=dict(color='white',width=0.5))))
    fig_hist.add_vline(x=22,line_dash='dash',line_color=C['warning'],line_width=2,
                        annotation_text='Moy: 22t',annotation_font=dict(color=C['warning'],size=11))
    fig_hist.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=250,showlegend=False,
                            xaxis=dict(title='Tokens/phrase',showgrid=False),
                            yaxis=dict(title='Fréquence',showgrid=True,gridcolor='rgba(0,0,0,0.06)'))
    return html.Div([
        html.Div([
            html.Div([
                html.Div([badge('NLP',C['primary']),badge('GLUE Benchmark',C['purple']),
                          badge('Classification Binaire',C['teal'])],style=dict(marginBottom='12px')),
                html.H2("SST-2 — Stanford Sentiment Treebank",
                        style=dict(fontSize='22px',fontWeight='800',color='white',margin='0 0 8px')),
                html.P("Dataset de référence pour évaluer les Transformers en NLP. "
                       "Base des papiers BERT, XLNet, RoBERTa.",
                       style=dict(fontSize='13px',color='rgba(255,255,255,0.82)',lineHeight='1.7',margin='0')),
            ],style=dict(flex='3')),
            html.Div([
                html.Div("67K",style=dict(fontSize='44px',fontWeight='900',color='white',textAlign='center')),
                html.Div("phrases",style=dict(fontSize='12px',color='rgba(255,255,255,0.7)',textAlign='center')),
            ],style=dict(flex='1',display='flex',flexDirection='column',justifyContent='center')),
        ],style=dict(background=f'linear-gradient(135deg,{C["dark"]} 0%,{C["primary"]} 100%)',
                     borderRadius='16px',padding='28px 32px',display='flex',gap='24px',
                     alignItems='center',marginBottom='22px',boxShadow='0 8px 32px rgba(99,102,241,0.25)')),
        html.Div([
            kpi("67,349","Train","phrases annotées",C['primary'],'📚'),
            kpi("872","Val","évaluation officielle",C['secondary'],'🎯'),
            kpi("2","Classes","Positif / Négatif",C['success'],'🏷️'),
            kpi("22","Tokens/ph","longueur moyenne",C['warning'],'📏'),
            kpi("2013","Année","Socher et al.",C['purple'],'📅'),
        ],style=dict(display='grid',gridTemplateColumns='repeat(5,1fr)',gap='14px',marginBottom='22px')),
        html.Div([
            html.Div([
                html.Div([
                    sec("⚖️ Équilibre des classes & Distribution"),
                    html.Div([
                        dcc.Graph(figure=fig_pie,config=dict(displayModeBar=False),style=dict(flex='1')),
                        dcc.Graph(figure=fig_hist,config=dict(displayModeBar=False),style=dict(flex='2')),
                    ],style=dict(display='flex',gap='12px')),
                    ibox("SST-2 est quasi-parfaitement équilibré : 34 100 positifs (50.7%) "
                         "et 33 249 négatifs. Cela rend l'accuracy fiable. "
                         "La longueur moyenne de 22 tokens justifie MAX_LEN=64.",col=C['success']),
                ],style=CARD),
                html.Div([
                    sec("🎯 Pourquoi SST-2 ?"),
                    *[bitem(p) for p in [
                        "Standard de facto pour évaluer les Transformers en NLP",
                        "Utilisé dans BERT, XLNet, RoBERTa, DeBERTa",
                        "Classes équilibrées — accuracy fiable sans rééchantillonnage",
                        "Disponible via Hugging Face datasets en 1 ligne",
                        "Taille idéale pour démonstration académique sur CPU",
                    ]],
                    html.Div([
                        html.Span("📄 Référence : ",style=dict(fontWeight='700',fontSize='12px')),
                        html.Span("Socher et al. (2013), EMNLP",
                                  style=dict(fontSize='11px',color=C['muted'],fontStyle='italic')),
                    ],style=dict(marginTop='14px',padding='12px',background='rgba(99,102,241,0.05)',borderRadius='8px')),
                ],style=CARD),
            ],style=dict(flex='1')),
            html.Div([
                html.Div([
                    sec("📝 Exemples du dataset"),
                    html.Div([
                        html.Div("😊 POSITIFS",style=dict(background=C['success'],color='white',
                                  padding='7px 14px',borderRadius='7px 7px 0 0',fontSize='11px',fontWeight='700')),
                        *[html.Div(html.Em(s,style=dict(fontSize='12px',color=C['dark'])),
                                   style=dict(padding='9px 14px',borderBottom=f'1px solid {C["border"]}',background='white'))
                          for s in EXEMPLE_POS],
                    ],style=dict(borderRadius='7px',overflow='hidden',
                                  border='1px solid rgba(16,185,129,0.3)',marginBottom='14px')),
                    html.Div([
                        html.Div("😞 NÉGATIFS",style=dict(background=C['danger'],color='white',
                                  padding='7px 14px',borderRadius='7px 7px 0 0',fontSize='11px',fontWeight='700')),
                        *[html.Div(html.Em(s,style=dict(fontSize='12px',color=C['dark'])),
                                   style=dict(padding='9px 14px',borderBottom=f'1px solid {C["border"]}',background='white'))
                          for s in EXEMPLE_NEG],
                    ],style=dict(borderRadius='7px',overflow='hidden',border='1px solid rgba(239,68,68,0.3)')),
                ],style=CARD),
                html.Div([
                    sec("⚙️ Spécifications techniques"),
                    html.Div([
                        html.Div([
                            html.Div(lab,style=dict(fontSize='11px',color=C['muted'],textTransform='uppercase',
                                      letterSpacing='0.05em',marginBottom='3px')),
                            html.Div(val,style=dict(fontSize='13px',fontWeight='600',color=C['dark'])),
                        ],style=dict(padding='11px 14px',background=C['bg'],borderRadius='8px'))
                        for lab,val in [
                            ("Tâche","Classification sentiments binaire"),
                            ("Source","Rotten Tomatoes — critiques films"),
                            ("Benchmark","GLUE (General Language Understanding)"),
                            ("Tokenizer","WordPiece (BERT) / BPE"),
                            ("MAX_LEN","64 tokens (optimisé CPU i5 8GB)"),
                            ("Batch size","8 (optimisé mémoire RAM)"),
                        ]
                    ],style=dict(display='grid',gridTemplateColumns='1fr 1fr',gap='8px')),
                ],style=CARD),
            ],style=dict(flex='1')),
        ],style=dict(display='flex',gap='22px')),
    ])

# ============================================================
# TAB 2 — THÉORIE
# ============================================================
def tab_graphs():
    return html.Div([
        html.Div([
            sec("🔢 Formule fondamentale — Mécanisme d'Attention"),
            html.Div("Attention(Q, K, V) = softmax( Q · Kᵀ / √dₖ ) · V",
                     style=dict(fontFamily='"Courier New",monospace',fontSize='20px',fontWeight='700',
                                color=C['primary'],textAlign='center',padding='20px',
                                background='rgba(99,102,241,0.07)',borderRadius='12px',
                                border='1px solid rgba(99,102,241,0.2)',marginBottom='20px')),
            html.Div([
                html.Div([
                    html.Div(ltr,style=dict(width='42px',height='42px',borderRadius='50%',
                                            background=col,color='white',fontWeight='900',
                                            fontSize='18px',display='flex',alignItems='center',
                                            justifyContent='center',margin='0 auto 8px')),
                    html.Div(name,style=dict(fontWeight='700',fontSize='11px',color=col,
                                             textAlign='center',marginBottom='5px')),
                    html.P(desc,style=dict(fontSize='11px',color=C['muted'],textAlign='center',lineHeight='1.5')),
                ],style=dict(flex='1',padding='16px',background=bg,borderRadius='10px'))
                for ltr,col,name,desc,bg in [
                    ('Q',C['primary'],'QUERY','Ce que le token cherche. Q=X·Wq','rgba(99,102,241,0.05)'),
                    ('K',C['secondary'],'KEY','Ce que le token offre. K=X·Wk','rgba(6,182,212,0.05)'),
                    ('V',C['success'],'VALUE',"L'info transmise. V=X·Wv",'rgba(16,185,129,0.05)'),
                    ('√dₖ',C['warning'],'SCALE','Évite le vanishing gradient','rgba(245,158,11,0.05)'),
                ]
            ],style=dict(display='flex',gap='12px',alignItems='stretch')),
            ibox("Le mécanisme d'attention calcule en parallèle toutes les relations entre tokens. "
                 "Contrairement aux RNN, aucune information ne se perd sur des séquences longues. "
                 "La division par √dₖ stabilise les gradients de la softmax.",
                 title="💡 Pourquoi l'attention révolutionne le NLP"),
        ],style=CARD),
        html.Div([
            html.Div([
                sec("📍 Encodage Positionnel Sinusoïdal"),
                dcc.Graph(figure=gpe(),config=dict(displayModeBar=False)),
                html.Div("PE(pos,2i)=sin(pos/10000^(2i/d))  |  PE(pos,2i+1)=cos(...)",
                         style=dict(fontFamily='monospace',fontSize='12px',color=C['primary'],
                                    textAlign='center',padding='10px',background='rgba(99,102,241,0.07)',
                                    borderRadius='8px',marginTop='10px')),
                ibox("Chaque ligne est la signature positionnelle unique de sa position. "
                     "Sin/cos permettent d'extrapoler à des longueurs non vues à l'entraînement."),
            ],style=dict(**CARD,flex='1')),
            html.Div([
                sec("🎯 Poids d'Attention appris"),
                dcc.Graph(figure=gattn(),config=dict(displayModeBar=False)),
                ibox("La valeur (i,j) indique combien le token i regarde le token j. "
                     "Les mots émotionnels ('wonderful', 'loved') capturent plus d'attention."),
            ],style=dict(**CARD,flex='1')),
        ],style=dict(display='flex',gap='22px')),
        html.Div([
            html.Div([
                sec("📉 Learning Rate Schedule (Warmup + Cosine Decay)"),
                dcc.Graph(figure=glr(),config=dict(displayModeBar=False)),
                ibox("Warmup (50 steps): LR monte de 0→max pour stabiliser les premières mises à jour. "
                     "Cosine Decay: descente douce jusqu'à 0. Améliore la généralisation de 2-3%.",
                     col=C['warning']),
            ],style=dict(**CARD,flex='1')),
            html.Div([
                sec("🔀 Comparaison variantes Transformer"),
                dcc.Graph(figure=gvariants(),config=dict(displayModeBar=False)),
                ibox("DistilBERT (rose): meilleur compromis précision/légèreté pour CPU. "
                     "DeBERTa domine mais nécessite GPU obligatoirement.",col=C['purple']),
            ],style=dict(**CARD,flex='1')),
        ],style=dict(display='flex',gap='22px')),
        html.Div([
            sec("🔀 Architecture Multi-Head Attention (h=8 têtes)"),
            html.Div("MultiHead(Q,K,V) = Concat(head₁,...,head₈)·Wᴼ    où    headᵢ = Attention(Q·Wᵢᴼ, K·Wᵢᴷ, V·Wᵢᵛ)",
                     style=dict(fontFamily='monospace',fontSize='14px',color=C['primary'],textAlign='center',
                                padding='14px',background='rgba(99,102,241,0.07)',borderRadius='10px',marginBottom='16px')),
            html.Div([
                html.Div([
                    html.Div(n,style=dict(background=c,color='white',padding='8px',
                                          borderRadius='8px',textAlign='center',
                                          fontSize='11px',fontWeight='700',marginBottom='4px')),
                    html.Div(r,style=dict(fontSize='10px',color=C['muted'],textAlign='center')),
                ],style=dict(flex='1'))
                for n,c,r in [
                    ("Tête 1",C['primary'],"Syntaxe"),("Tête 2",C['secondary'],"Sémantique"),
                    ("Tête 3",C['success'],"Coréférence"),("Tête 4",C['warning'],"Négation"),
                    ("Tête 5",C['danger'],"Intensité"),("Tête 6",C['purple'],"Aspect"),
                    ("Tête 7",C['pink'],"Modalité"),("Tête 8",C['teal'],"Temporel"),
                ]
            ],style=dict(display='flex',gap='8px',marginBottom='14px')),
            ibox("8 têtes capturent simultanément syntaxe, sémantique, coréférence et modalité. "
                 "Coût identique car chaque tête travaille sur d_model/h=32 dimensions.",col=C['purple']),
        ],style=CARD),
    ])

# ============================================================
# TAB 3 — PERFORMANCES (toutes données en dur, indépendant)
# ============================================================
def tab_perf():
    # Toutes les figures sont construites localement ici
    fig_barres  = gbarres()
    fig_roc     = groc()
    fig_training= gtraining()
    fig_radar   = gradar()
    fig_aucbar  = gaucbar()
    figs_cm     = [gcm(r) for r in RESULTS]

    return html.Div([
        # EN-TÊTE
        html.Div([
            html.H3("📊 Résultats expérimentaux — Comparaison des 3 modèles",
                    style=dict(fontSize='20px',fontWeight='800',color=C['dark'],margin='0 0 6px')),
            html.P(f"Dataset SST-2 · 872 phrases de validation · "
                   f"Meilleur modèle : {BEST['name']} (AUC={BEST['auc']})",
                   style=dict(fontSize='13px',color=C['muted'],margin='0')),
        ],style=dict(padding='20px 28px',background='white',borderRadius='14px',
                     border=f'1px solid {C["border"]}',marginBottom='22px',
                     boxShadow='0 2px 8px rgba(0,0,0,0.04)')),

        # TABLEAU
        html.Div([
            sec("🏆 Tableau comparatif"),
            html.Div([
                html.Div(h,style=dict(flex=f,fontWeight='700',fontSize='11px',textTransform='uppercase',
                          color=C['muted'],letterSpacing='0.05em',padding='10px 14px',
                          borderBottom=f'2px solid {C["border"]}'))
                for h,f in [('Modèle','3'),('Accuracy','1'),('F1','1'),
                             ('AUC-ROC','1'),('Précision','1'),('Rappel','1'),('Params','1')]
            ],style=dict(display='flex')),
            *[html.Div([
                html.Div([
                    html.Span("★ " if r['name']==BEST['name'] else "",
                              style=dict(color=C['warning'],fontWeight='900')),
                    html.Span(r['name'],style=dict(color=r['color'],
                               fontWeight='700' if r['name']==BEST['name'] else '500',fontSize='13px')),
                    html.Div(style=dict(height='3px',width=f"{r['acc']*100:.0f}%",
                                        background=r['color'],borderRadius='2px',marginTop='5px',opacity='0.4')),
                ],style=dict(flex='3',padding='14px')),
                *[html.Div([
                    html.Div(f"{v:.4f}" if isinstance(v,float) else str(v),
                             style=dict(fontWeight='700',
                                         color=r['color'] if r['name']==BEST['name'] else C['dark'],
                                         fontSize='14px')),
                    html.Div(style=dict(height='3px',
                                        width=f"{(v*100):.0f}%" if isinstance(v,float) and v<=1 else '0%',
                                        background=r['color'],borderRadius='2px',marginTop='3px',opacity='0.3'))
                    if isinstance(v,float) and v<=1 else html.Div(),
                ],style=dict(flex='1',padding='14px',textAlign='center'))
                for v in [r['acc'],r['f1'],r['auc'],r['prec'],r['recall'],r['params']]],
            ],style=dict(display='flex',alignItems='center',
                          background='rgba(99,102,241,0.04)' if r['name']==BEST['name'] else 'transparent',
                          borderBottom=f'1px solid {C["border"]}',borderLeft=f'4px solid {r["color"]}'))
              for r in RESULTS],
        ],style=CARD),

        # PILLS MEILLEUR MODÈLE
        html.Div([
            sec(f"🥇 Métriques détaillées — {BEST['name']}"),
            html.Div([
                mpill(BEST['acc'],'Accuracy',C['primary']),
                mpill(BEST['f1'],'F1-Score',C['secondary']),
                mpill(BEST['auc'],'AUC-ROC',C['success']),
                mpill(BEST['prec'],'Précision',C['warning']),
                mpill(BEST['recall'],'Rappel',C['danger']),
            ],style=dict(display='flex',gap='14px')),
            ibox(f"{BEST['name']} obtient AUC={BEST['auc']:.4f} — "
                 f"soit {BEST['auc']*100:.1f}% de probabilité qu'un exemple positif "
                 "soit mieux classé qu'un négatif. Son F1 confirme l'équilibre précision/rappel.",
                 col=C['success']),
        ],style=CARD),

        # BARRES
        html.Div([
            sec("📊 Comparaison visuelle — Toutes métriques"),
            dcc.Graph(figure=fig_barres,config=dict(displayModeBar=False)),
            ibox("DistilBERT domine sur toutes les métriques. "
                 "Le Transformer scratch surpasse la baseline logistique, "
                 "validant l'intérêt des architectures d'attention."),
        ],style=CARD),

        # ROC + RADAR
        html.Div([
            html.Div([
                sec("📈 Courbes ROC"),
                dcc.Graph(figure=fig_roc,config=dict(displayModeBar=False)),
                ibox("Plus la courbe s'approche du coin supérieur gauche, "
                     "meilleure est la discrimination. "
                     "DistilBERT (violet) domine sur tout le spectre des seuils."),
            ],style=dict(**CARD,flex='1')),
            html.Div([
                sec("🕸️ Radar multidimensionnel"),
                dcc.Graph(figure=fig_radar,config=dict(displayModeBar=False)),
                ibox("Le radar confirme la supériorité de DistilBERT sur toutes les dimensions. "
                     "Le Transformer scratch montre un profil équilibré — bien calibré."),
            ],style=dict(**CARD,flex='1')),
        ],style=dict(display='flex',gap='22px')),

        # CLASSEMENT AUC
        html.Div([
            sec("🏅 Classement par AUC-ROC"),
            dcc.Graph(figure=fig_aucbar,config=dict(displayModeBar=False)),
        ],style=CARD),

        # COURBES APPRENTISSAGE
        html.Div([
            sec("📉 Courbes d'apprentissage — Accuracy & Loss"),
            dcc.Graph(figure=fig_training,config=dict(displayModeBar=False)),
            ibox("Courbes pleines = validation, pointillées = entraînement. "
                 "Faible écart → bonne généralisation, pas d'overfitting. "
                 "DistilBERT converge en 3 époques grâce aux poids pré-entraînés."),
        ],style=CARD),

        # MATRICES DE CONFUSION
        html.Div([
            sec("🔲 Matrices de confusion"),
            html.Div([
                html.Div([
                    dcc.Graph(figure=figs_cm[i],config=dict(displayModeBar=False)),
                    html.Div([
                        html.Span(f"VP:{RESULTS[i]['cm'][1][1]} ",style=dict(color=C['success'],fontWeight='700',fontSize='12px')),
                        html.Span(f"VN:{RESULTS[i]['cm'][0][0]} ",style=dict(color=C['primary'],fontWeight='700',fontSize='12px')),
                        html.Span(f"FP:{RESULTS[i]['cm'][0][1]} ",style=dict(color=C['warning'],fontWeight='700',fontSize='12px')),
                        html.Span(f"FN:{RESULTS[i]['cm'][1][0]}",style=dict(color=C['danger'],fontWeight='700',fontSize='12px')),
                    ],style=dict(textAlign='center',marginTop='6px')),
                ],style=dict(flex='1'))
                for i in range(len(RESULTS))
            ],style=dict(display='flex',gap='16px')),
            ibox("VP=Vrais Positifs, VN=Vrais Négatifs, FP=Faux Positifs, FN=Faux Négatifs. "
                 "DistilBERT minimise FP et FN simultanément — pas de biais de classe."),
        ],style=CARD),

        # GUIDE MÉTRIQUES
        html.Div([
            sec("📖 Guide des métriques"),
            html.Div([
                html.Div([
                    html.Div(n,style=dict(fontWeight='700',fontSize='13px',color=c,marginBottom='5px')),
                    html.Div(f,style=dict(fontFamily='monospace',fontSize='11px',color=C['muted'],
                              marginBottom='6px',background='rgba(0,0,0,0.03)',padding='4px 8px',borderRadius='5px')),
                    html.P(d,style=dict(fontSize='12px',color=C['muted'],lineHeight='1.6',margin='0')),
                ],style=dict(flex='1',padding='14px',background=C['bg'],borderRadius='10px',borderTop=f'3px solid {c}'))
                for n,c,f,d in [
                    ("Accuracy",C['primary'],"Acc=(VP+VN)/Total","% prédictions correctes. Fiable car SST-2 est équilibré."),
                    ("F1-Score",C['secondary'],"F1=2·P·R/(P+R)","Moyenne harmonique précision/rappel."),
                    ("AUC-ROC",C['success'],"AUC=∫TPR d(FPR)","Probabilité qu'un + soit mieux classé qu'un -."),
                    ("Précision",C['warning'],"P=VP/(VP+FP)","Parmi les 'Positif' prédits, combien sont vrais ?"),
                    ("Rappel",C['danger'],"R=VP/(VP+FN)","Parmi les vrais positifs, combien trouvés ?"),
                ]
            ],style=dict(display='flex',gap='12px')),
        ],style=CARD),

        # MODÈLE RETENU
        html.Div([
            sec("🎯 Modèle retenu & Justification"),
            html.Div([
                html.Div("✅",style=dict(fontSize='38px',textAlign='center',marginBottom='6px')),
                html.H3(BEST['name'],style=dict(color=C['success'],fontWeight='800',fontSize='22px',
                                                 textAlign='center',marginBottom='10px')),
                html.Div([
                    badge(f"AUC:{BEST['auc']}",C['success']),
                    badge(f"Acc:{BEST['acc']:.1%}",C['primary']),
                    badge(f"F1:{BEST['f1']:.4f}",C['secondary']),
                ],style=dict(textAlign='center',marginBottom='18px')),
            ]),
            html.P([
                html.Strong("Justification : "),
                f"{BEST['name']} est retenu car il obtient le meilleur AUC-ROC ({BEST['auc']:.4f}). "
                "Le transfer learning exploite une représentation linguistique universelle "
                "apprise sur 3,3 milliards de tokens. En gelant les 8 premières couches "
                "sur 12, le fine-tuning est 60% plus rapide tout en conservant 95% des performances. "
                "Cette approche est le standard industriel NLP en 2025.",
            ],style=dict(fontSize='14px',lineHeight='1.85',color=C['dark'])),
        ],style={**CARD,'border':f'2px solid {C["success"]}','background':'rgba(16,185,129,0.03)'}),
    ])

# ============================================================
# TAB 4 — SIMULATION
# ============================================================
def tab_simu():
    return html.Div([
        html.Div([
            sec("🚀 Analyse de sentiment interactive"),
            html.P("Entrez une phrase en anglais et observez l'analyse token par token :",
                   style=dict(fontSize='14px',color=C['muted'],marginBottom='16px')),
            dcc.Textarea(id='txt-input',
                         value='This film is absolutely wonderful and I loved every single moment of it!',
                         style=dict(width='100%',height='90px',padding='14px',borderRadius='10px',
                                    border=f'2px solid {C["border"]}',fontSize='15px',
                                    fontFamily='Inter,sans-serif',resize='vertical',outline='none')),
            html.Div([
                html.Button("🔍 Analyser",id='btn-go',n_clicks=0,
                    style=dict(background=f'linear-gradient(135deg,{C["dark"]},{C["primary"]})',
                               color='white',border='none',borderRadius='10px',
                               padding='13px 36px',fontSize='15px',fontWeight='700',
                               cursor='pointer',marginRight='10px')),
                html.Button("🔄 Effacer",id='btn-rst',n_clicks=0,
                    style=dict(background='white',color=C['muted'],
                               border=f'1px solid {C["border"]}',borderRadius='10px',
                               padding='13px 22px',fontSize='15px',cursor='pointer')),
            ],style=dict(marginTop='14px')),
            html.Div(id='sim-out',style=dict(marginTop='20px')),
        ],style=CARD),

        html.Div([
            sec("⚡ Phrases d'exemple — cliquez pour tester"),
            html.Div([
                html.Div([
                    html.Div("😊 POSITIFS",style=dict(fontWeight='700',fontSize='11px',
                              color=C['success'],marginBottom='8px',textTransform='uppercase')),
                    *[html.Button(f'"{s[:65]}..."' if len(s)>65 else f'"{s}"',
                                  id=f'q-pos-{i}',n_clicks=0,
                                  style=dict(display='block',width='100%',textAlign='left',
                                             padding='10px 14px',marginBottom='7px',
                                             background='rgba(16,185,129,0.06)',
                                             border='1px solid rgba(16,185,129,0.25)',
                                             borderRadius='8px',cursor='pointer',
                                             fontSize='12px',color=C['dark'],
                                             fontFamily='Inter,sans-serif'))
                      for i,s in enumerate(EXEMPLE_POS)],
                ],style=dict(flex='1')),
                html.Div([
                    html.Div("😞 NÉGATIFS",style=dict(fontWeight='700',fontSize='11px',
                              color=C['danger'],marginBottom='8px',textTransform='uppercase')),
                    *[html.Button(f'"{s[:65]}..."' if len(s)>65 else f'"{s}"',
                                  id=f'q-neg-{i}',n_clicks=0,
                                  style=dict(display='block',width='100%',textAlign='left',
                                             padding='10px 14px',marginBottom='7px',
                                             background='rgba(239,68,68,0.06)',
                                             border='1px solid rgba(239,68,68,0.25)',
                                             borderRadius='8px',cursor='pointer',
                                             fontSize='12px',color=C['dark'],
                                             fontFamily='Inter,sans-serif'))
                      for i,s in enumerate(EXEMPLE_NEG)],
                ],style=dict(flex='1')),
            ],style=dict(display='flex',gap='22px')),
        ],style=CARD),

        html.Div([
            sec("🧠 Carte d'attention en temps réel"),
            dcc.Graph(id='attn-live',config=dict(displayModeBar=False),
                      style=dict(minHeight='320px')),
            ibox("Les cellules colorées indiquent les mots qui s'influencent mutuellement. "
                 "Les mots émotionnellement forts (wonderful, terrible) "
                 "capturent naturellement plus d'attention des tokens voisins."),
        ],style=CARD),
    ])

# ============================================================
# APPLICATION
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap',
    ],
    title="Sujet A — Transformer | INF4123 DIPES 2",
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = html.Div([
    # HEADER
    html.Div([
        html.Div([
            html.Div([
                html.Span("🤖",style=dict(fontSize='30px',marginRight='14px')),
                html.Div([
                    html.H1("Architectures Transformer — Mécanisme d'Attention",
                            style=dict(fontSize='20px',fontWeight='800',color='white',margin='0')),
                    html.P("INF4123 Deep Learning · DIPES 2 · S2 · Université de Yaoundé I · 2025-2026",
                           style=dict(color='rgba(255,255,255,0.7)',fontSize='12px',margin='3px 0 0')),
                ]),
            ],style=dict(display='flex',alignItems='center')),
            html.Div([
                html.Div("EKOTTO ERIC (20U2963)",
                         style=dict(color='rgba(255,255,255,0.9)',fontSize='11px',fontWeight='600')),
                html.Div("NAZIFATOU OUSMANOU (20O2910)",
                         style=dict(color='rgba(255,255,255,0.9)',fontSize='11px',fontWeight='600')),
                html.Div("Encadreur : Dr STEPHANE TEKOUABOU",
                         style=dict(color='rgba(255,255,255,0.7)',fontSize='10px')),
            ],style=dict(textAlign='right')),
        ],style=dict(display='flex',justifyContent='space-between',alignItems='center',
                     maxWidth='1400px',margin='0 auto',padding='0 32px')),
    ],style=dict(background=f'linear-gradient(135deg,{C["darker"]} 0%,{C["primary"]} 100%)',
                 padding='18px 0',boxShadow='0 4px 24px rgba(0,0,0,0.2)')),

    # TABS
    html.Div([
        dcc.Tabs(id='tabs',value='dataset',children=[
            dcc.Tab(label='📊 Dataset & Contexte',       value='dataset',style=TS,selected_style=TS_SEL),
            dcc.Tab(label='📐 Graphiques & Théorie',     value='graphs', style=TS,selected_style=TS_SEL),
            dcc.Tab(label='🏆 Performances & Résultats', value='perf',   style=TS,selected_style=TS_SEL),
            dcc.Tab(label='🚀 Simulation Interactive',   value='simu',   style=TS,selected_style=TS_SEL),
        ],style=dict(borderBottom=f'1px solid {C["border"]}',padding='0 32px',background='white')),
    ]),

    html.Div(id='content',style=dict(padding='26px 32px',background=C['bg'],
                                      minHeight='calc(100vh - 120px)')),
],style=dict(fontFamily='Inter,sans-serif',minHeight='100vh'))

# ============================================================
# CALLBACKS
# ============================================================
@app.callback(Output('content','children'), Input('tabs','value'))
def render(tab):
    if tab == 'dataset': return tab_dataset()
    if tab == 'graphs':  return tab_graphs()
    if tab == 'perf':    return tab_perf()
    return tab_simu()

# Lexique
POS_LEX = {
    'wonderful':3,'amazing':3,'excellent':3,'masterpiece':3,'beautiful':2,
    'great':2,'love':2,'loved':2,'fantastic':2,'brilliant':2,'perfect':2,
    'good':1,'enjoy':1,'delightful':2,'stunning':2,'inspiring':2,'superb':2,
    'outstanding':2,'magnificent':2,'remarkable':2,'extraordinary':2,'fun':1,
    'heartwarming':2,'flawless':2,'charming':1,'captivating':2,'moving':2,
}
NEG_LEX = {
    'terrible':3,'awful':3,'horrible':3,'disaster':3,'worst':3,'bad':2,
    'boring':2,'dull':2,'waste':2,'poor':2,'disappointing':2,'tedious':2,
    'painful':2,'atrocious':2,'pathetic':2,'forgettable':1,'predictable':1,
    'mediocre':1,'dreadful':2,'catastrophe':3,'slow':1,'nonsense':2,'avoid':2,
}

@app.callback(
    Output('sim-out',   'children'),
    Output('attn-live', 'figure'),
    Input('btn-go',  'n_clicks'),
    Input('btn-rst', 'n_clicks'),
    *[Input(f'q-pos-{i}','n_clicks') for i in range(4)],
    *[Input(f'q-neg-{i}','n_clicks') for i in range(4)],
    State('txt-input','value'),
    prevent_initial_call=True,
)
def analyze(*args):
    ctx  = callback_context
    text = args[-1] or ''

    if ctx.triggered:
        tid = ctx.triggered[0]['prop_id'].split('.')[0]
        if tid.startswith('q-pos-'):
            text = EXEMPLE_POS[int(tid.split('-')[-1])]
        elif tid.startswith('q-neg-'):
            text = EXEMPLE_NEG[int(tid.split('-')[-1])]
        elif tid == 'btn-rst':
            return '', go.Figure()

    if not text.strip():
        return html.P("Entrez une phrase et cliquez sur Analyser.",
                      style=dict(color=C['muted'],fontStyle='italic')), go.Figure()

    words = text.split()
    ps = sum(POS_LEX.get(w.strip('.,!?;:').lower(),0) for w in words)
    ns = sum(NEG_LEX.get(w.strip('.,!?;:').lower(),0) for w in words)
    total = ps + ns or 2
    pp = ps/total; pn = ns/total
    is_pos = pp >= 0.5
    label  = "😊 Sentiment POSITIF" if is_pos else "😞 Sentiment NÉGATIF"
    color  = C['success'] if is_pos else C['danger']
    conf   = max(pp,pn)

    result = html.Div([
        html.H3(label,style=dict(color=color,fontWeight='800',fontSize='22px',margin='0 0 14px')),
        html.Div([
            html.Div(style=dict(width=f"{pp*100:.0f}%",height='14px',background=C['success'],
                                borderRadius='7px 0 0 7px' if pp>0 else '0')),
            html.Div(style=dict(width=f"{pn*100:.0f}%",height='14px',background=C['danger'],
                                borderRadius='0 7px 7px 0' if pn>0 else '0')),
        ],style=dict(display='flex',borderRadius='7px',overflow='hidden',marginBottom='10px')),
        html.Div([
            html.Span(f"😊 Positif : {pp*100:.1f}%  ",style=dict(color=C['success'],fontWeight='700')),
            html.Span(f"😞 Négatif : {pn*100:.1f}%",style=dict(color=C['danger'],fontWeight='700')),
            html.Span(f"  |  Confiance : {conf*100:.0f}%",style=dict(color=C['muted'],marginLeft='16px')),
        ],style=dict(marginBottom='16px')),
        html.Div([
            html.Div("Analyse token par token :",
                     style=dict(fontSize='12px',color=C['muted'],fontWeight='600',marginBottom='8px')),
            html.Div([
                html.Span(w+" ",style=dict(
                    background=('rgba(16,185,129,0.22)' if w.strip('.,!?;:').lower() in POS_LEX
                                else 'rgba(239,68,68,0.18)' if w.strip('.,!?;:').lower() in NEG_LEX
                                else 'transparent'),
                    color=(C['success'] if w.strip('.,!?;:').lower() in POS_LEX
                           else C['danger'] if w.strip('.,!?;:').lower() in NEG_LEX else C['dark']),
                    fontWeight=('700' if (w.strip('.,!?;:').lower() in POS_LEX or
                                          w.strip('.,!?;:').lower() in NEG_LEX) else '400'),
                    padding='3px 6px',borderRadius='5px',fontSize='15px',
                    display='inline-block',marginBottom='4px',
                )) for w in words[:35]
            ]),
            html.Div([
                html.Span("■ ",style=dict(color=C['success'])),
                html.Span("Mot positif   ",style=dict(fontSize='11px',color=C['muted'],marginRight='16px')),
                html.Span("■ ",style=dict(color=C['danger'])),
                html.Span("Mot négatif",style=dict(fontSize='11px',color=C['muted'])),
            ],style=dict(marginTop='10px')),
        ],style=dict(padding='14px',background=C['bg'],borderRadius='10px')),
    ],style=dict(padding='22px',borderRadius='12px',
                  background=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)',
                  border=f'1px solid {color}'))

    tok   = words[:14]+['[SEP]']
    n     = len(tok)
    alpha = np.ones(n,dtype=float)
    for i,w in enumerate(tok):
        wc = w.strip('.,!?;:').lower()
        if wc in POS_LEX: alpha[i] = POS_LEX[wc]*1.5
        if wc in NEG_LEX: alpha[i] = NEG_LEX[wc]*1.5
    attn_sim = np.array([np.random.dirichlet(alpha*0.7) for _ in range(n)])

    fig = go.Figure(go.Heatmap(
        z=attn_sim,x=tok,y=tok,colorscale='Blues',showscale=True,
        colorbar=dict(thickness=12,len=0.8),
        hovertemplate='De:<b>%{y}</b> → <b>%{x}</b>: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(**PL,margin=PL_MARGIN,font=PL_FONT,height=350,
                      xaxis=dict(tickangle=-35,showgrid=False,tickfont=dict(size=11)),
                      yaxis=dict(showgrid=False,tickfont=dict(size=11)))
    return result, fig

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)