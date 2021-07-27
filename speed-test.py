def f1(a,b,c,d):
  return (a + b + d) % 430

def f2(a,b,c,d):
  return (3*a + c) % 430

def f3(a,b,c,d):
  return (4*a + d) % 430

def f4(a,b,c,d):
  return (2*b + c) % 430
 
def f5(a,b,c,d):
  return (4*a + 2*b + 2*c + d) % 430 

def check(f1,f2,f3,f4,f5):
  if f1 == 3 and f2 == 6 and f3 == 13 and f4 == 26 and f5 == 5:
    return True
  else:
    return False
  
 for a in range(130,430):
  print("a:" , a)
  for b in range(0,430):
    for c in range(0,430):
      for d in range(0,430):
        y1 = f1(a,b,c,d)
        y2 = f2(a,b,c,d)
        y3 = f3(a,b,c,d)
        y4 = f4(a,b,c,d)
        y5 = f5(a,b,c,d)
        if check(y1,y2,y3,y4,y5) == True:
          print('Solution!', str(a), str(b), str(c), str(d)) 
