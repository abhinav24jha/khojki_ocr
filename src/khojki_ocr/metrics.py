"""Code-point Levenshtein metrics with a reproducible edit backtrace."""
def alignment(reference, prediction):
    n,m=len(reference),len(prediction)
    d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0]=i
    for j in range(m+1): d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(reference[i-1]!=prediction[j-1]))
    ops=[]; i,j=n,m
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(reference[i-1]!=prediction[j-1]):
            ops.append(('equal' if reference[i-1]==prediction[j-1] else 'substitute',reference[i-1],prediction[j-1],i-1,j-1)); i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:
            ops.append(('delete',reference[i-1],'',i-1,j));i-=1
        else:
            ops.append(('insert','',prediction[j-1],i,j-1));j-=1
    return list(reversed(ops))

def score(reference,prediction):
    ops=alignment(reference,prediction)
    counts={k:sum(o[0]==k for o in ops) for k in ['substitute','delete','insert']}
    errors=sum(counts.values())
    return {'reference_characters':len(reference),'errors':errors,'cer':errors/len(reference) if reference else (0 if not prediction else None),**counts}
