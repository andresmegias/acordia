#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Acordia.

This script produces guitar chord charts for the options stored in the input
configuration file.

Andrés Megías
"""

config_file = 'config.yaml'

import sys
import copy
import itertools
import yaml
import numpy as np
import matplotlib.pyplot as plt

def relu(x, a=0):
    y = np.maximum(a, x)
    return y

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def model(x, W):

    W1 = W[0]
    b1 = W[1]
    W2 = W[2]
    b2 = W[3]
    W3 = W[4]
    b3 = W[5]
    W4 = W[6]
    b4 = W[7]
    
    a1 = relu(np.dot(W1.T, x) + b1)
    a2 = relu(np.dot(W2.T, a1) + b2)
    a3 = relu(np.dot(W3.T, a2) + b3)
    a4 = sigmoid(np.dot(W4.T, a3) + b4)
    y = a4[0]
    
    return y

def positions(D):
    x = np.zeros(D.shape[0], int)
    inds = D.nonzero()
    for i in range(len(inds)):
        x[inds[0]] = inds[1]
    return x

def binarize(x, l=0.5):
    y = np.zeros(len(x), int)
    for i in range(len(x)):
        if x[i] > l:
            y[i] = 1
    return y

def equivalent_positions(x):
    v = np.array(x)
    cond = v != 0
    if v.sum() != 0:
        d1 = v[cond].min() - 1
    else:
        d1 = 0
    u = np.zeros(len(v), int)
    u[cond] = v[cond] - d1
    return u

def separator(A):
    Li, Lj = A.shape
    V = [[] for i in range(Li)]
    for i in range(Li):
        if A[i,:].max() == 0:
            v = np.zeros(Lj,int)
            V[i] += [v]
        else:
            for j in range(Lj):
                if A[i,j] == True:
                    v = np.zeros(Lj,int)
                    v[j] = A[i,j] 
                    V[i] += [v]
    B = []
    for U in itertools.product(*V):
        M = np.zeros((Li,Lj),int)
        for i in range(Li):
            M[i,:] = U[i]
        B += [M]
    return B

def fret_distance(n, show_realistic_frets=True):
    if show_realistic_frets:
        f = 2**(1/12)
        f = 1 / (1 - 1/f)
        dist = f * (1 - 1/2**(n/12))
    else:
        dist = n
    return dist

def step(x, s=1):
    y = x * (s * np.sign(x) + 1) / 2
    return y

def plot_diagrs(diagrs, ps, h0=0, colors=['mediumblue','steelblue']):

    color1, color2 = colors
    nh = 15
    
    if len(diagrs) != 0:
        Nh = len(diagrs)//nh+1
    else:
        Nh = 0
    
    for h in range(Nh):      
        
        plt.figure(2+h+h0, figsize=(14,9))
        plt.clf()
        plt.subplots_adjust(left=marg, right=1-marg, top=1-marg, bottom=marg,
                            wspace=0.4, hspace=0.4)
         
        m = 0
        
        for diagr, p in zip(diagrs[h*nh:(h+1)*nh] ,ps[h*nh:(h+1)*nh]):
                
            m += 1
            
            sketch = diagr[:,1:].nonzero()
            sketch1 = (diagr[:,1:]==2).astype(int).nonzero()
            
            x = sketch[0] + 1
            y = sketch[1]
            x1 = sketch1[0] + 1
            y1 = sketch1[1]
            
            notes_text = []
            for i in range(len(x)):
                ind_diagr = int(octave.index(strings[x[i]-1][:-1])+y[i]+p)%12
                for nota in chord_notes:
                    num_sharps = nota.count('#')
                    num_flats = nota.count('b')
                    ind_chord = octave.index(nota[0]) + num_sharps - num_flats
                    if octave[ind_diagr] == octave[ind_chord]:
                        nota_i = nota.replace('b','♭').replace('#','♯')
                        notes_text += [nota_i]
        
            notes_text = np.array(notes_text)
            
            y = -fret_distance(y+Nt+p, show_realistic_frets)
            y1 = -fret_distance(y1+Nt+p, show_realistic_frets)
            
            sp = plt.subplot(3,5,m)
            
            np_ = diagr.shape[1]-1
            for i in range(0,np_+1):
                plt.hlines(-fret_distance(i+Nt+p-0.5, show_realistic_frets),
                           fx*1, fx*Nc, color='k')
            for i in range(Nc):
                plt.vlines(fx*(i+1), -fret_distance(np_+Nt+p-0.5,
                           show_realistic_frets),
                           -fret_distance(Nt+p-0.5, show_realistic_frets),
                           color='k') 
    
            plt.plot(fx*x[y!=0],y[y!=0], '.', ms=ms, color=color2,
                     clip_on=False)
            plt.plot(fx*x[y==0],y[y==0], '.', ms=ms, color=color2,
                     clip_on=False)
    
            plt.plot(fx*x1[y1!=0], y1[y1!=0], '.', ms=ms, color=color1,
                     clip_on=False)
            plt.plot(fx*x1[y1==0], y1[y1==0], '.', ms=ms, color=color1,
                     clip_on=False)       
                           
            #-0.3*int(Nt!=0)
            if show_note_names:
                for i in range(len(y!=0)):
                    plt.text(fx*x[y!=0][i], y[y!=0][i]-dy, notes_text[y!=0][i],
                             color='white', horizontalalignment='center',
                             verticalalignment='center', fontsize=fs)
            
            sketch = diagr[:,0].nonzero()
            sketch1 = (diagr[:,0]==2).astype(int).nonzero()
            
            x = sketch[0] + 1
            y = 0*x
            x1 = sketch1[0] + 1
            y1 = 0*x1
            
            notes_text = []
            for i in range(len(x)):
                ind_diagr = int(octave.index(strings[x[i]-1][:-1]))%12
                for note in chord_notes:
                    num_sharps = note.count('#')
                    num_flats = note.count('b')
                    ind_chord = octave.index(note[0]) + num_sharps - num_flats
                    if octave[ind_diagr] == octave[ind_chord]:
                        note_i = note.replace('b','♭').replace('#','♯')
                        notes_text += [note_i]
            notes_text = np.array(notes_text)
            
            y = -fret_distance(y+Nt+p-1, show_realistic_frets)
            y1 = -fret_distance(y1+Nt+p-1, show_realistic_frets)
            
            plt.plot(fx*x1, y1, '.', ms=ms, color=color1, clip_on=False)            
            plt.plot(fx*x, y, '.', ms=ms, color=color2, clip_on=False)   
            plt.plot(fx*1, -fret_distance(0+Nt+p-1,show_realistic_frets),
                     '.', ms=ms, color='red', alpha=0, clip_on=False)

            if show_note_names and len(strings) <= 6:
                for i in range(len(y)):
                    plt.text(fx*x[i], y[i]-dy, notes_text[i], color='white',
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=fs)
            
            xlocs = fx * np.linspace(1,Nc,Nc)
            xlabels = config['string notes'].split(',')
            for i in range(Nc):
                xlabels[i] = ' ' + xlabels[i][:-1] + '$_' + xlabels[i][-1] + '$'
                xlabels[i] = (xlabels[i].replace(',','').replace('b','♭')
                             .replace('#','♯'))
            
            locspos = [Nt]
            for i in range(1, np_+1):
                locspos += [i+p-1+Nt]  
            locspos = np.array(locspos)
            ylocs = -0.5-fret_distance(locspos-0.5, show_realistic_frets)
            ylocs[0] = -fret_distance(locspos[1]-1, show_realistic_frets)
            ylabels = np.array(locspos, str)
            if ylabels[0] == '0':
                ylabels[0] = 'o'
            plt.yticks(ylocs, ylabels)        
            
            plt.xticks(xlocs, xlabels)
        
            if (diagr != 0).astype(int).sum(axis=1)[0] != 0:
                pady = 10
            else:
                pady = 6
            plt.tick_params(axis='x', pad=6)
            plt.tick_params(axis='y', pad=pady)
            
            #plt.ylim([ylocs.min(), ylocs.max()])
            plt.axis('scaled')
            #plt.axis('off')
            
            if pers:
                fw = 'regular'
            else:
                fw = 'bold'
            
            plt.title(title.replace('b','$♭$').replace('#','$♯$'),
                      fontweight=fw,pad=10)
            
            plt.tick_params(top=False, bottom=False, left=False, right=False)
            
            for ax in ['top','bottom','left','right']:
                sp.spines[ax].set_visible(False)
    
        if m == 0:
            plt.close(2+h)
    
        if save_figures:
            if Nh == 1:
                numtable = ''
            else:
                numtable = '_' + str(h+1)
            plt.figure(h+2)
            plt.savefig('table-'+ figname + numtable + '.' + img_format,
                        dpi=200)

#%% Loading of the configuration file and setting up.

if len(sys.argv) == 2:
    config_file = sys.argv[1]

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

default_options = {
        'strings': ['E2,A2,D3,G3,B3,E4'],
        'number of frets': 12,
        'capo fret': 0,
        'show realistic frets': True,
        'show redundant chord combinations': False,
        'show chords with 4 frets': True,
        'use neural network': True,
        'show note names': True,
        'image format': '.png',
        'save images': False
        }

config = {**default_options, **config}

strings = config['string notes']
Np = config['number of frets']
Nt = config['capo fret']
chord = config['chord']
show_realistic_frets = config['show realistic frets']
show_redundant_chords = config['show redundant chord combinations']
use_neural_network = config['use neural network']
show_note_names = config['show note names']
img_format = config['image format']
save_figures = config['save images']
show_chords_4frets = config['show chords with 4 frets']

if len(strings.split(',')) != 6:
    use_neural_network = False
        
strings = strings.split(',') 

octave = ['E','F','F#','G','G#','A','A#','B','C','C#','D','D#']
scale = ['E','F','G','A','B','C','D']
     
notes = []      
Nn = 5   
for i in range(Nn):
    notes += octave

ind = 1   
for note in notes:
    if note == 'C':
        ind += 1
    notes[notes.index(note)] += str(ind)

Nc = len(strings)
Np += 1
frets = [[[] for p in range(Np)] for c in range(Nc)]

for c in range(Nc):
    ind = notes.index(strings[c].replace('#','').replace('b',''))
    ind += strings[c].count('#') - strings[c].count('b') + Nt
    frets[c][0] = notes[ind]
    strings[c] = notes[ind]
       
for c in range(Nc):
    for p in range(Np):
        frets[c][p] = notes[notes.index(frets[c][0])+p]

#%% Reading of the chord.

title = copy.copy(chord)

if chord[0] == '(' and chord[-1] == ')':
    pers = True
    chord = chord[1:-1]
    chord = chord.split(',')
    chord_notes = copy.copy(chord)
    for note in chord:
        num_sharps = note.count('#')
        num_flats = note.count('b')
        num_alterations = num_sharps + num_flats
        ind = octave.index(note[0]) + num_sharps - num_flats
        chord[chord.index(note)] = octave[ind]
else:
    pers = False
    num_sharps = chord.count('#')
    num_flats = chord.count('b')
    num_alterations = num_sharps + num_flats
    ind = octave.index(chord[0]) + num_sharps - num_flats
    semitones = []
    if len(chord) ==  1+num_alterations+1 and chord[-1] == 'm':
        semitones = [3,7]
        grades = [3,5]
    elif len(chord) > 1+num_alterations and chord[1+num_alterations] == '5':
        semitones = [7]
        grades = [5]
    elif len(chord) > 1+num_alterations and chord[1+num_alterations] == '7':
        semitones = [4,7,10]  
        grades = [3,5,7]
    elif 'M7' in chord:
        semitones = [4,7,11]
        grades = [3,5,7]
    elif 'm7' in chord:
        semitones = [3,7,10]
        grades = [3,5,7]
    elif 'dim' in chord:
        semitones = [3,6]
        grades = [3,5]
    elif 'aug' in chord:
        semitones = [4,8]
        grades = [3,5]
    else:
        semitones = [4,7]
        grades = [3,5]
  
    grades = np.array(grades)
    semitones = np.array(semitones)
    major_scale = np.array([2,4,5,7,9,11,12])
    semitones_ref = major_scale[grades-2]
    
    chord = [[] for i in range(1+len(semitones))]
    chord_notes = copy.copy(chord)
    chord[0] = octave[ind]
    chord_notes[0] = title[0]
    for i in range(num_flats):
        chord_notes[0] += 'b'
    for i in range(num_sharps):
        chord_notes[0] += '#'
    for i in range(len(semitones)):
        chord[i+1] = octave[(ind+semitones[i])%12]
        chord_notes[i+1] = scale[(scale.index(chord_notes[0][0])+grades[i]-1)%7]
        diff_alt = octave.index(chord_notes[i+1]) - octave.index(chord[i+1]) 
        if abs(diff_alt) > 6:
            diff_alt = np.sign(diff_alt) * (abs(diff_alt)-12)
        diff_flats = int(abs(step(diff_alt, 1)))
        diff_sharps = int(abs(step(diff_alt, -1)))
        for j in range(diff_flats):
            chord_notes[i+1] += 'b'
        for j in range(diff_sharps):
            chord_notes[i+1] += '#'
        chord_notes[i+1] = chord_notes[i+1].replace('#b','').replace('b#','')

show_first_note = True

marks = np.zeros((Nc,Np))
marks[:,0] = 0

for c in range(Nc):
    for p in range(Np):
        for note in chord:
            if note in frets[c][p] and note+'#' not in frets[c][p]:
                marks[c][p] = 1
                if show_first_note and note == chord[0]:
                    marks[c][p] = 2

#%% Plot of the entire bridge.

fx = 0.6  # horizontal size factor
ms = 40  # narkersize
fs = 9  # fontsize
dy = 0.02  # vertical offset

color1 = 'mediumblue'
color2 = 'steelblue'

sketch = marks.nonzero()
sketch1 = (marks==2).astype(int).nonzero()
L = len(sketch[0])

x = sketch[0] + 1
y = sketch[1] + Nt
x1 = sketch1[0] + 1
y1 = sketch1[1] + Nt


notes_text = []
for i in range(len(x)):
    ind_diagr = int(octave.index(strings[x[i]-1][:-1])+y[i]-Nt)%12
    for note in chord_notes:
        num_sharps = note.count('#')
        num_flats = note.count('b')
        ind_chord = octave.index(note[0]) + num_sharps - num_flats
        if octave[ind_diagr] == octave[ind_chord]:
            note_i = note.replace('b','♭').replace('#','♯')
            notes_text += [note_i]

notes_text = np.array(notes_text)

y = -fret_distance(y, show_realistic_frets)
y1 = -fret_distance(y1, show_realistic_frets)


figsize = (0.6*(2+fx*6),
           0.6*(3 + fret_distance(Nt+Np, show_realistic_frets)
                - fret_distance(Nt, show_realistic_frets)))
plt.figure(1, figsize=figsize)
plt.clf()
sp = plt.subplot(1,1,1)

for i in range(0,Np):
    plt.hlines(-fret_distance(0.5+i+Nt,show_realistic_frets), fx*1, fx*Nc,
               color='k')
for i in range(Nc):
    plt.vlines(fx*(i+1), -fret_distance(Np+Nt-0.5,show_realistic_frets),
               -fret_distance(Nt+0.5,show_realistic_frets), color='k')    
    
plt.plot(fx*x[y!=0], y[y!=0], '.', ms=ms, color=color2, clip_on=False)
plt.plot(fx*x[y==0], y[y==0], '.', ms=ms, color=color2, clip_on=False)

plt.plot(fx*x1[y1!=0], y1[y1!=0], '.', ms=ms, color=color1, clip_on=False)
plt.plot(fx*x1[y1==0], y1[y1==0], '.', ms=ms, color=color1, clip_on=False)

plt.plot(fx*1, -fret_distance(Nt) + 0.02, '.', ms=ms, color='red', alpha=0,
         clip_on=False)


if show_note_names:
    for i in range(len(y)):
        plt.text(fx*x[i], y[i]-dy, notes_text[i], color='white',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=fs)


xlocs = fx*np.linspace(1,Nc,Nc)
xlabels = copy.copy(strings)
for i in range(Nc):
    xlabels[i] = ' ' + xlabels[i][:-1] + '$_' + xlabels[i][-1] + '$'
    xlabels[i] = xlabels[i].replace(',','').replace('b','♭').replace('#','♯')

nums = [5,7,12]
locspos = [Nt]
for num in nums:
    if num > Nt and num < Nt+Np:
        locspos += [num]
ylabels = locspos
locspos = np.array(locspos)
ylocs = -fret_distance(locspos, show_realistic_frets)
ylabels = np.array(locspos, str)
if ylabels[0] == '0':
    ylabels[0] = 'o'
    
#sp.xaxis.tick_top()
plt.xticks(xlocs, xlabels)
plt.yticks(ylocs, ylabels)
plt.margins(x=0, y=0)
plt.tick_params(axis='x', pad=12)
plt.tick_params(axis='y', pad=18)

if pers:
    fw = 'regular'
else:
    fw = 'bold'

plt.title(title.replace('b','♭').replace('#','♯'), fontweight=fw, pad=20)
      
plt.axis('scaled')
#plt.axis('off')

plt.tick_params(top=False, bottom=False, left=False, right=False)

for eje in ['top','bottom','left','right']:
    sp.spines[eje].set_visible(False)

if Nt == 0:
    namecapo = ''
else:
    namecapo = '-(+' + str(Nt) + ')'
figname = '('+','.join(strings)+')' + namecapo + '-' + title

if save_figures:
    plt.savefig(figname + '.' +  img_format, dpi=200)

#%% Calculation of the bar charts.

nps = [3,4]
Nac = 4
if len(chord) <= 2 or Nc <= 4:
    Nac = 3

c1 = []
p1 = []
N1 = 0
Ncmax = 3
for c in range(Ncmax):
    for p in range(Np):
        if marks[c,p] == 2:
            N1 += 1
            c1 += [c]
            p1 += [p]
    
diagrs1 = []
ps1 = []
     
for n in range(N1):
    
    for np_ in nps:
    
        pmin = max(1, 1+p1[n]-np_)
        pmax = min(Np-np_, p1[n])
        
        if p1[n] == 0:
            pmax = Np - np_
              
        for p in range(pmin, pmax+1):
                
            diagr = np.zeros((Nc,np_+1))
            
            for k in range(Nc):
                if k != c1[n]:
                    diagr[k,0] = marks[k,0]
            
            if p1[n] == 0:
                diagr[c1[n],0] = 2
            else:
                diagr[c1[n],p1[n]-p+1] = 2
            
            pl = min(Np, p+np_) - p
            diagr[:,np_+1-pl:] = marks[:,p:p+pl]    
            
            ind1 = notes.index(strings[c1[n]]) + p1[n]  
            cd = diagr.nonzero()[0]
            pd = diagr.nonzero()[1]
            for i in range(len(cd)):
                if pd[i] == 0:
                    ind = notes.index(strings[cd[i]])
                else:
                    ind = notes.index(strings[cd[i]]) + pd[i] + p-1
                if cd[i] != c1[n] and ind < ind1:
                    diagr[cd[i],pd[i]] = 0
                if cd[i] == c1[n] and ind != ind1:
                    diagr[cd[i],pd[i]] = 0
            
            Ntp = (((diagr != 0).astype(int).sum(axis=1))!=0).sum()
            if Ntp >= Nac:
                diagrs1 += [diagr]
                ps1 += [p]
        
diagrs2 = []
ps2 = []
 
for d in range(len(diagrs1)):
    conds = separator((diagrs1[d] != 0))
    for cond in conds:
        diagrs2 += [diagrs1[d]*cond]
        ps2 += [ps1[d]]

diagrs3 = []
ps3 = []
    
for d in range(len(diagrs2)):
    np_ = diagrs2[d].shape[1]-1
    sketch = (diagrs2[d] != 0).astype(int)
    sketch_string = sketch.sum(axis=1)
    if sketch_string.sum() >= Nac:
        for c in range(Nc):
            if sketch_string[c] == 1:
                iters = 1 + Nc - c - Nac
                for a in range(iters):
                    diagr = np.zeros((Nc, np_+1))
                    diagr[c:Nc-a,:] = diagrs2[d][c:Nc-a,:]
                    if sketch_string[c:Nc-a].sum() >= Nac:
                        diagrs3 += [diagr]
                        ps3 += [ps2[d]]
                break
            
repeateds = []                   
for di in range(len(diagrs3)):
    for dj in range(di+1, len(diagrs3)):
        if np.array_equal(diagrs3[di], diagrs3[dj]) and dj not in repeateds:
            repeateds += [dj]

diagrs4 = []
ps4 = []          
for d in range(len(diagrs3)):
    if d not in repeateds:
        diagrs4 += [diagrs3[d]]
        ps4 += [ps3[d]]

valspos = []
for d in range(len(diagrs4)):
    valspos += [diagrs4[d].nonzero()[1]] 
    
repeateds = []
for di in range(len(diagrs4)):
    condi = valspos[di] != 0
    for dj in range(di+1,len(diagrs4)):
        condj = valspos[dj] != 0
        if np.array_equal(condi,condj):
            difij = valspos[dj][condj] - valspos[di][condi]
            dpij = - (ps4[dj] - ps4[di])
            if (np.array_equal(difij, dpij*np.ones(len(difij)))
                    and dj not in repeateds):
                repeateds += [dj]
                    
diagrs5 = []
ps5 = []  
for d in range(len(diagrs4)):
    if d not in repeateds:
        diagrs5 += [diagrs4[d]]
        ps5 += [ps4[d]]

arpeggios = False
diagrs6 = []
ps6 = []
for d in range(len(diagrs5)):
    sketch = (diagrs5[d] != 0).astype(int)
    sketch_string = sketch.sum(axis=1)
    i1 = sketch_string.nonzero()[0][0]
    i2 = sketch_string.nonzero()[0][-1]
    if np.array_equal(sketch_string[i1:i2+1], np.ones(1+i2-i1)) or arpeggios:
        diagrs6 += [diagrs5[d]]
        ps6 += [ps5[d]]
    elif len(chord) <= 2:
        np_ = diagrs5[d].shape[1]-1
        diagr = np.zeros((Nc, np_+1))
        i2 = i1+2
        if np.array_equal(sketch_string[i1:i2+1], np.ones(1+i2-i1)):
            diagr[i1:i2+1,:] = diagrs5[d][i1:i2+1,:]
            diagrs6 += [diagr]
            ps6 += [ps5[d]]

no = []
for d in range(len(diagrs6)):
    notes_diagr = []
    cd = diagrs6[d].nonzero()[0]
    pd = diagrs6[d].nonzero()[1]
    for i in range(len(cd)):
        if pd[i] != 0:
            ind = int(octave.index(strings[cd[i]][:-1])+pd[i]+ps6[d]-1)%12
            notes_diagr += [octave[ind]]
        else:
            ind = int(octave.index(strings[cd[i]][:-1])+pd[i])%12
            notes_diagr += [octave[ind]]
    
    for note in chord:
        if note not in notes_diagr and d not in no:
            no += [d]

diagrs7 = []
ps7 = []  

for d in range(len(diagrs6)):
    if d not in no:
        diagrs7 += [diagrs6[d]]
        ps7 += [ps6[d]]

diagrs8 = []
ps8 = []
for d in range(len(diagrs7)):
    if np.array_equal(diagrs7[d][:,1], np.zeros(Nc)):
        np_ = diagrs7[d].shape[1]-1
        diagr = np.zeros((Nc,np_+1))
        diagr[:,0] = diagrs7[d][:,0]
        diagr[:,1:-1] = diagrs7[d][:,2:]
        p = ps7[d] + 1
    else:
        diagr = copy.copy(diagrs7[d])
        p = ps7[d]
    diagrs8 += [diagr]
    ps8 += [p]


diagrs9 = []
ps9 = []
for d in range(len(diagrs8)):
    sketch = (diagrs8[d]!=0).astype(int)
    Ntp = sketch[:,1:].sum()
    if not np.array_equal(sketch[:,0], np.zeros(Nc)):
        ind0 = sketch[:,0].nonzero()[0][0]
    else:
        ind0 = -1
    if not np.array_equal(sketch[:,1], np.zeros(Nc)):
        ind1 = sketch[:,1].nonzero()[0][0]
    else:
        ind1 = -1
    if ind0 < ind1:
        Ntp -= sketch[:,1].sum() - 1
    if Ntp <= 4:
        diagrs9 += [diagrs8[d]]
        ps9 += [ps8[d]]
        

if show_chords_4frets:
    dplim = 4
else:
    dplim = 3


diagrs10 = []
ps10 = []

for d in range(len(diagrs9)):
    
    diagr = np.zeros((Nc,max(nps)+1))
    np_ = diagrs9[d].shape[1]
    diagr[:,:np_] = diagrs9[d][:,:np_]
    p = copy.copy(ps9[d])
    pos = (diagr!=0).astype(int).nonzero()[1]
    if pos.sum() != 0:
        posmin = pos[pos!=0].min() - 1
    else:
        posmin = 0
    posmax = pos.max()
    d_1 = ps9[d] - 1
    d_pmax = 4 - posmax

    if posmax <= dplim:
        
        if d_1 != 0 and d_1 <= d_pmax:
            
            diagr = np.zeros((Nc, max(nps)+1))
            diagr[:,0] = diagrs9[d][:,0]
            diagr[:,(1+d_1):] = diagrs9[d][:,1:(5-d_1)]
            p -= d_1

        elif d_pmax != 0 and d_1 > d_pmax:
            
            diagr = np.zeros((Nc,max(nps)+1))
            diagr[:,0] = diagrs9[d][:,0]
            diagr[:,1:(4-posmin)] = diagrs9[d][:,(1+posmin):]
            p += posmin
        diagrs10 += [diagr]
        ps10 += [p]
       
diagrs11 = []
ps11 = []
y10 = []
if use_neural_network:
    weights = np.load('weights.npy', allow_pickle=True)
    for d in range(len(diagrs10)):
        x = equivalent_positions(positions(diagrs10[d]))
        y = binarize([model(x, weights)])[0]
        y10 += [y]

repeateds = []
for di in range(len(diagrs10)):
    cd = diagrs10[di].nonzero()
    Ni = (diagrs10[di].sum(axis=1)!=0).sum()
    if Ni <= 5:
        ind1 = cd[0][0]
        ind2 = cd[0][-1]
        posi = (diagrs10[di]!=0).astype(int)[ind1:ind2+1,1:].nonzero()
        for dj in range(len(diagrs10)):
            Nj = (diagrs10[dj].sum(axis=1)!=0).sum()
            posj = (diagrs10[dj]!=0).astype(int)[ind1:ind2+1,1:].nonzero()
            if dj != di and Nj > Ni and np.array_equal(posi[0], posj[0]):
                dyij = posj[1] - posi[1]
                dpij = -(ps10[dj] - ps10[di])
                cond1 = np.array_equal(diagrs10[di][ind1:ind2+1,0],
                                       diagrs10[dj][ind1:ind2+1,0])
                cond2 = np.array_equal(dyij, dpij*np.ones(len(dyij)))
                if di < len(y10) and dj < len(y10):
                    cond3 = y10[di] == y10[dj]
                else:
                    cond3 = False
                
                if cond1 and cond2 and cond3 and di not in repeateds:
                    repeateds += [di]
  
diagrs11 = []
ps11 = []  
y11 = []          
for d in range(len(diagrs10)):
    if d not in repeateds or show_redundant_chords:
        diagrs11 += [diagrs10[d]]
        ps11 += [ps10[d]]
        if use_neural_network:
            y11 += [y10[d]]

inds = np.argsort(ps11)

diagrs12 = [diagrs11[i] for i in inds]
ps12 = [ps11[i] for i in inds]


if use_neural_network:
    y12 = np.array([y11[i] for i in inds])
    inds1 = (y12 == 1).nonzero()[0]
    diagrs13 = [diagrs12[i] for i in inds1]
    ps13 = [ps12[i] for i in inds1]
    inds0 = (y12 == 0).nonzero()[0]
    diagrs_no = [diagrs12[i] for i in inds0]
    ps_no = [ps12[i] for i in inds0]  
else:
    diagrs13 = copy.copy(diagrs12)
    ps13 = copy.copy(ps12)

diagrs = copy.copy(diagrs13)
ps = copy.copy(ps13)


#%% Plots of the bar charts.

fx = 0.75  # horizontal scale factor
marg = 0.07  # margin
ms = 34  # markersize
fs = 8  # fontsize

color1 = 'mediumblue'
color2 = 'steelblue'

plot_diagrs(diagrs, ps)

nh = 15
Nh = len(diagrs) // nh+1

show_discarded_chords = False
if use_neural_network and show_discarded_chords:
    plot_diagrs(diagrs_no, ps_no, h0=Nh, colors=['dimgray','gray'])   

plt.show()
