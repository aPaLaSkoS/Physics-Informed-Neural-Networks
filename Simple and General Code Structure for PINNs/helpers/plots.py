def plot_real(x,y,F):
  F_xt = F
  fig,ax=plt.subplots(1,1)

  cp = ax.contourf(x,y, F_xt,20,cmap="rainbow")
  fig.colorbar(cp)
  ax.set_title(r'''
  Homogeneous 2D wave equation

  $ u_{real}(x,y) $
  ''')
  ax.set_xlabel(r'$x$')
  ax.set_ylabel(r'$y$')
  plt.show()

  ax = plt.axes(projection='3d')
  ax.plot_surface(x, y, F_xt,cmap="rainbow")
  ax.set_title(r'''
  Homogeneous 2D wave equation

  $ u_{real}(x,y) $
  ''')
  ax.set_xlabel(r'$x$')
  ax.set_ylabel(r'$y$')
  plt.show()
  

def plot_pred(x,y,F):
  F_xt = F
  fig,ax=plt.subplots(1,1)

  cp = ax.contourf(x,y, F_xt,20,cmap="rainbow")
  fig.colorbar(cp)
  ax.set_title(r'''
  Homogeneous 2D wave equation

  $ u_{pred}(x,y) $
  ''')
  ax.set_xlabel(r'$x$')
  ax.set_ylabel(r'$y$')
  plt.show()

  ax = plt.axes(projection='3d')
  ax.plot_surface(x, y, F_xt,cmap="rainbow")
  ax.set_title(r'''
  Homogeneous 2D wave equation

  $ u_{pred}(x,y) $
  ''')
  ax.set_xlabel(r'$x$')
  ax.set_ylabel(r'$y$')
  plt.show()


def plot_abs_diff(x,y,F):
  F_xt = F
  fig,ax=plt.subplots(1,1)

  cp = ax.contourf(x,y, F_xt,20,cmap="rainbow")
  fig.colorbar(cp)
  ax.set_title(r'''
  Homogeneous 2D wave equation

  $ |u_{pred}(x,y) - u_{real}(x,y)|$
  ''')
  ax.set_xlabel(r'$x$')
  ax.set_ylabel(r'$y$')
  plt.show()

  ax = plt.axes(projection='3d')
  ax.plot_surface(x, y, F_xt,cmap="rainbow")
  ax.set_title(r'''
  Homogeneous 2D wave equation

  $ |u_{pred}(x,y) - u_{real}(x,y)|$
  ''')
  ax.set_xlabel(r'$x$')
  ax.set_ylabel(r'$y$')
  plt.show()