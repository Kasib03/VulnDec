# myutils_minimal.py

import keyword
import builtins
from keras import backend as K
import tensorflow as tf

def stripComments(code):
    lines = code.split("\n")
    cleaned = ""
    for line in lines:
        if "#" in line:
            line = line[:line.find("#")]
        cleaned += line + "\n"
    return cleaned

def getBadpart(change):
    lines = change.split("\n")
    badexamples = []
    goodexamples = []
    for line in lines:
        line = line.lstrip()
        if len(line.replace(" ", "")) > 1:
            if line.startswith("-") and not line[1:].lstrip().startswith("#"):
                badexamples.append(line[1:])
            elif line.startswith("+") and not line[1:].lstrip().startswith("#"):
                goodexamples.append(line[1:])
    return [badexamples, goodexamples] if badexamples else None

def findposition(badpart, sourcecode):
    b = badpart.strip()
    if not b:
        return [-1, -1]
    start = sourcecode.find(b)
    end = start + len(b) if start != -1 else -1
    return [start, end] if start != -1 else [-1, -1]

def findpositions(badparts, sourcecode):
    return [findposition(bp, sourcecode) for bp in badparts if findposition(bp, sourcecode) != [-1, -1]]

def nextsplit(code, pos):
    splitchars = set(" \t\n.:()[]<>+-=\"'*\\~{}!?;,%%&")
    for i in range(pos + 1, len(code)):
        if code[i] in splitchars:
            return i
    return -1

def previoussplit(code, pos):
    splitchars = set(" \t\n.:()[]<>+-=\"'*\\~{}!?;,%%&")
    for i in range(pos - 1, -1, -1):
        if code[i] in splitchars:
            return i
    return -1

def getcontextPos(code, focus, fulllength):
    if focus >= len(code): return None
    start, end = focus, focus
    while end - start < fulllength:
        ps = previoussplit(code, start)
        ns = nextsplit(code, end)
        if ps == -1 and ns == -1: break
        if ps != -1: start = ps
        if ns != -1: end = ns
    return [start, end] if end - start >= 1 else None

def getblocks(code, badpositions, step, fulllength):
    blocks = []
    focus = 0
    lastfocus = 0
    while focus < len(code):
        if code[lastfocus:focus] != "\n":
            mid = lastfocus + (focus - lastfocus) // 2
            context = getcontextPos(code, mid, fulllength)
            if context:
                label = any((context[0] <= b[0] <= context[1]) or (context[0] <= b[1] <= context[1]) for b in badpositions)
                snippet = code[context[0]:context[1]]
                if not any(snippet == b[0] for b in blocks):
                    blocks.append([snippet, 1 if label else 0])
        if "\n" in code[focus + 1:focus + 7]:
            lastfocus = focus
            focus += code[focus + 1:focus + 7].find("\n") + 1
        else:
            ns = nextsplit(code, focus + step)
            if ns != -1:
                lastfocus = focus
                focus = ns
            else:
                break
    return blocks

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def f1(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    ap = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = tp / (pp + K.epsilon())
    recall = tp / (ap + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


