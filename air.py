import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animmation
from numpy import poly1d
import sympy as sp  
# s=np.array([[0,round(np.random.random()*100),round(np.random.random()*100),round(np.random.random()*100),
# round(np.random.random()*100)],[0,round(np.random.random()*100),round(np.random.random()*100),round(np.random.random()*100),round(np.random.random()*100)],[0,0,0,0,0]],dtype=int) 
s=np.array([[0,0,100,0,100],[0,0,0,100,100],[0,0,0,0,0]],dtype=int)   #s初始位置 
a=np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=int)   #相关矩阵

#h = np.array([[-15,-5,15,5],[-30,-30,-30,-30],[0,0,0,0]],dtype=int) #偏移矩阵
h = np.array([[15,-15,-15,15],[-15,15,-15,15],[0,0,0,0]],dtype=int)
#h = np.array([[-13,-8,13,8],[-15,-35,-15,-35],[0,0,0,0]],dtype=int)
#h = np.array([[-15,15,-7,7],[-20,-20,-10,-10],[0,0,0,0]],dtype=int)

x0 = 50 + 30 * sp.cos(2  * sp.symbols('t'))
y0 = 50 + 30 * sp.sin(2  * sp.symbols('t'))
z0=50+ 30 * sp.sin(2  * sp.symbols('t'))
# x0 = 50 
# y0 = 50 
# z0=50
Fx1=0
Fy1=0
Fz1=0
Fx2=0
Fy2=0
Fz2=0
Fx3=0
Fy3=0
Fz3=0
Fx4=0
Fy4=0
Fz4=0
def update(data):
    #print(data)
    line0.set_data([data[0], data[1]])
    line0.set_3d_properties(data[2])
    line1.set_data([data[3], data[4]])
    line1.set_3d_properties(data[5])
    line2.set_data([data[6], data[7]])
    line2.set_3d_properties(data[8])
    line3.set_data([data[9], data[10]])
    line3.set_3d_properties(data[11])
    line4.set_data([data[12], data[13]])
    line4.set_3d_properties(data[14])
    return line0,line1,line2,line3,line4

def data_gen():
    global x0,y0,z0,Fx1,Fy1,Fz1,Fx2,Fy2,Fz2,Fx3,Fy3,Fz3,Fx4,Fy4,Fz4
    data = []
    fx0 = sp.lambdify(sp.symbols('t'), x0)
    fy0 = sp.lambdify(sp.symbols('t'), y0)
    fz0 = sp.lambdify(sp.symbols('t'), z0)
    fx1, Fx1=getFunc(1, s[0][1],  'x')
    fy1, Fy1=getFunc(1, s[1][1],  'y')
    fz1, Fz1=getFunc(1, s[2][1],  'z')
    fx2, Fx2=getFunc(2, s[0][2],  'x')
    fy2, Fy2=getFunc(2, s[1][2],  'y')
    fz2, Fz2=getFunc(2, s[2][2],  'z')
    fx3, Fx3=getFunc(3, s[0][3],  'x')
    fy3, Fy3=getFunc(3, s[1][3],  'y')
    fz3, Fz3=getFunc(3, s[2][3],  'z')
    fx4, Fx4=getFunc(4, s[0][4],  'x')
    fy4, Fy4=getFunc(4, s[1][4],  'y')
    fz4, Fz4=getFunc(4, s[2][4],  'z')
    t_range = np.arange(0, 100, 0.01)
    t_len = len(t_range)
    for ti in range(1,t_len):
        t = t_range[ti]
        data.append([fx0(t), fy0(t), fz0(t), fx1(t), fy1(t), fz1(t), fx2(t), fy2(t), fz2(t), fx3(t), fy3(t), fz3(t), fx4(t), fy4(t), fz4(t)])
    return data

def getFunc(k, s, p):
    if p=='x':
        f1=sp.dsolve(sp.diff(f(t),t,2)-sp.diff(x0,t,2) +  a[0][k-1]*(f(t)-  x0 -h[0][k-1] +2*(sp.diff(f(t),t,1)-sp.diff(x0,t,1)) )
         +a[1][k-1]*(f(t)-h[0][k-1] -(Fx1-h[0][0]) +2*(sp.diff(f(t),t,1)-sp.diff(Fx1,t,1)))
         +a[2][k-1]*(f(t)-h[0][k-1] -(Fx2-h[0][1]) +2*(sp.diff(f(t),t,1)-sp.diff(Fx2,t,1)))  
         +a[3][k-1]*(f(t)-h[0][k-1] -(Fx3-h[0][2]) +2*(sp.diff(f(t),t,1)-sp.diff(Fx3,t,1)))  
         +a[4][k-1]*(f(t)-h[0][k-1] -(Fx4-h[0][3]) +2*(sp.diff(f(t),t,1)-sp.diff(Fx4,t,1))),f(t))
    if p=='y':
        f1=sp.dsolve(sp.diff(f(t),t,2)-sp.diff(y0,t,2) +  a[0][k-1]*(f(t)-  y0 -h[1][k-1] +2*(sp.diff(f(t),t,1)-sp.diff(y0,t,1)) )
         +a[1][k-1]*(f(t)-h[1][k-1] -(Fy1-h[1][0]) +2*(sp.diff(f(t),t,1)-sp.diff(Fy1,t,1)))
         +a[2][k-1]*(f(t)-h[1][k-1] -(Fy2-h[1][1]) +2*(sp.diff(f(t),t,1)-sp.diff(Fy2,t,1)))  
         +a[3][k-1]*(f(t)-h[1][k-1] -(Fy3-h[1][2]) +2*(sp.diff(f(t),t,1)-sp.diff(Fy3,t,1)))  
         +a[4][k-1]*(f(t)-h[1][k-1] -(Fy4-h[1][3]) +2*(sp.diff(f(t),t,1)-sp.diff(Fy4,t,1))),f(t))
    if p=='z':
        f1=sp.dsolve(sp.diff(f(t),t,2)-sp.diff(z0,t,2) +  a[0][k-1]*(f(t)-  z0 -h[2][k-1] +2*(sp.diff(f(t),t,1)-sp.diff(z0,t,1)) )
         +a[1][k-1]*(f(t)-h[2][k-1] -(Fz1-h[2][0]) +2*(sp.diff(f(t),t,1)-sp.diff(Fz1,t,1)))
         +a[2][k-1]*(f(t)-h[2][k-1] -(Fz2-h[2][1]) +2*(sp.diff(f(t),t,1)-sp.diff(Fz2,t,1)))  
         +a[3][k-1]*(f(t)-h[2][k-1] -(Fz3-h[2][2]) +2*(sp.diff(f(t),t,1)-sp.diff(Fz3,t,1)))  
         +a[4][k-1]*(f(t)-h[2][k-1] -(Fz4-h[2][3]) +2*(sp.diff(f(t),t,1)-sp.diff(Fz4,t,1))),f(t))
    f1=f1.rhs
    f2=sp.diff(f1, t)
    solve=sp.solve([f1.subs(t,0)-s, f2.subs(t,0)],[C1, C2])
    f1= f1.subs({C1:solve[C1],C2:solve[C2]})
    print(str(k)+"号机"+p+"控制函数："+str(f1))
    f_func = sp.lambdify(t, f1)
    return f_func, f1

	
if __name__=="__main__":
 
    t=sp.symbols('t')
    f=sp.Function('f')
    C1=sp.symbols('C1')
    C2=sp.symbols('C2')
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    line0, = ax.plot([s[0][0]], [s[1][0]], [s[2][0]], marker='^', color='r', markersize=12)
    line1, = ax.plot([s[0][1]], [s[1][1]], [s[2][1]], marker='^', color='b', markersize=8)
    line2, = ax.plot([s[0][2]], [s[1][2]], [s[2][2]], marker='^', color='b', markersize=8)
    line3, = ax.plot([s[0][3]], [s[1][3]], [s[2][3]], marker='^', color='b', markersize=8)
    line4, = ax.plot([s[0][4]], [s[1][4]], [s[2][4]], marker='^', color='b', markersize=8)
    ani = animmation.FuncAnimation(fig, update, frames = data_gen(),  interval = 10)
    plt.show()
