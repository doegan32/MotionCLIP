import numpy as np

import matplotlib.animation 
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


#import Animation



def GetSkeletonInformation(skeletonName, scale=1):

    if skeletonName == "Edinburgh":
        parents = [-1,  0,  1,  2,  3,  4,  2,  6,  7,  8 , 9 , 2 ,11 ,12, 13, 14,  0, 16, 17, 18,  0 ,20 ,21 ,22,  0 ,24, 25]
        offsets = np.array(
            [[  0.  ,        0.       ,   0.       ],
                [  0.     ,     0.        ,  0.       ],
                [ 19.     ,     0.        ,  0.       ],
                [ 22.5    ,     0.6       ,  0.       ],
                [ 14.     ,     0.0308777 ,  0.       ],
                [ 17.     ,     0.        ,  0.       ],
                [ 19.8    ,     3.7       ,  4.3      ],
                [  8.     ,     0.        ,  0.       ],
                [ 15.2    ,     0.        ,  0.       ],
                [ 17.8    ,     0.        ,  0.       ],
                [  7.2    ,     0.        ,  0.       ],
                [ 19.8    ,     3.7       , -4.3      ],
                [  8.     ,     0.        ,  0.151654 ],
                [ 15.2    ,     0.        ,  0.       ],
                [ 17.8    ,     0.        ,  0.       ],
                [  7.2    ,     0.        ,  0.       ],
                [  5.98425,    -7.666     ,  4.78879  ],
                [ 16.     ,     0.        ,  0.       ],
                [ 18.     ,     0.        ,  0.       ],
                [  0.     ,   -10.8       ,  0.       ],
                [  5.98425,    -7.66598   , -4.78879  ],
                [ 16.     ,     0.        ,  0.       ],
                [ 18.     ,     0.        ,  0.       ],
                [  0.     ,   -10.8       ,  0.       ],
                [  6.83696,    -0.722574  ,  0.       ],
                [ 12.     ,     0.        ,  0.       ],
                [ 12.     ,     0.        ,  0.       ]]
        )
    elif skeletonName == "PFNN":
        parents = [-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18, 15, 20, 21, 22, 23, 24, 25, 23, 27, 15, 29, 30, 31, 32, 33, 34, 32, 36]
        offsets = np.array(
            [[0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],  
            [1.3631, -1.7946, 0.8393], 
            [2.4481, -6.7261, 0.0000], 
            [2.5622, -7.0396, 0.0000], 
            [0.1576, -0.4331, 2.3226], 
            [0.0000, 0.0000, 0.0000],  
            [0.0000, 0.0000, 0.0000],  
            [-1.3055, -1.7946, 0.8393],
            [-2.5425, -6.9855, 0.0000],
            [-2.5683, -7.0562, 0.0000],
            [-0.1647, -0.4526, 2.3632],
            [0.0000, 0.0000, 0.0000],  
            [0.0000, 0.0000, 0.0000],  
            [0.0283, 2.0356, -0.1934], 
            [0.0567, 2.0488, -0.0428], 
            [0.0000, 0.0000, 0.0000],  
            [-0.0542, 1.7462, 0.1720], 
            [0.1041, 1.7614, -0.1240], 
            [0.0000, 0.0000, 0.0000],  
            [0.0000, 0.0000, 0.0000],  
            [3.3624, 1.2009, -0.3112], 
            [4.9830, -0.0000, -0.0000],
            [3.4836, -0.0000, -0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.7153, -0.0000, -0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [-3.1366, 1.3740, -0.4047],
            [-5.2419, -0.0000, -0.0000],
            [-3.4442, -0.0000, -0.0000],
            [0.0000, 0.0000, 0.0000],
            [-0.6225, -0.0000, -0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000]
                ]
        )
    elif skeletonName == "amass":
        parents = [-1, 0, 1, 2, 3, 4, 1, 6 , 7, 8, 1, 10, 11, 12, 13, 14, 15, 16, 12, 18, 19, 20, 21, 12, 23]  
        offsets = 0
    else:
        assert False, 'GetSkeletonInformation, unknown skeletonName'

    return parents, offsets*scale


# helper function to make using colours easier
def ColourNameToNum():
	d = {'red': (255,0,0), 'green': (0,255,0), 'blue':(0,0,255), 'yellow':(255,255,0), 'cyan':(0,255,255), 'magenta':(255,0,255), 'purple':(127,0,127), 'green_dark':(0,127,127), 'yellow_dark':(255, 201, 14)}
	return d

# helper function to enable plotting different body parts in different colours
# All hardcoded for the different skeletons we use, i.e. Bath quadruped, Edinburgh quadruped, Vicon humanoid
def GetDefaultColours(skeletonName):

    colours = ColourNameToNum()
    jointColours = []

    if skeletonName == "Bath":
        jointColours = [
            colours['red'], colours['red'], colours['red'],
            colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'],
            colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'],
            colours['green'], colours['green'], colours['green'], colours['green'], colours['green'],
            colours['green_dark'], colours['green_dark'], colours['green_dark'], colours['green_dark'], colours['green_dark'],
            colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue'],
            colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
            colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark']
            ]
    elif skeletonName == "Edinburgh":
        jointColours = [
            colours['red'], colours['red'], colours['red'],
            colours['green'], colours['green'],
            colours['green_dark'],
            colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'],
            colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'],
            colours['blue'], colours['blue'], colours['blue'], colours['blue'],
            colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
            colours['yellow_dark'], colours['yellow_dark'], colours['yellow_dark']]
    elif skeletonName == "PFNN":
        jointColours = [
            colours['red'],
            colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue'],
            colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
            colours['red'], colours['red'], colours['red'],
            colours['green'], colours['green'],
            colours['green_dark'], colours['green_dark'],
            colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'],
            colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple']
            ]
    elif skeletonName == "Human":
        jointColours = [
            colours['red'], colours['red'], colours['red'], colours['red'], colours['red'],
            colours['green'], colours['green'], colours['green'],
            colours['green_dark'], 
            colours['purple'], colours['purple'], colours['purple'], colours['yellow'], colours['purple'], colours['purple'], colours['purple'], colours['yellow'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'], 
            colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], 
            colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
            colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue']
        ]
    elif skeletonName == "HumanNoHands":
        jointColours = [colours['red'], colours['red'], colours['red'], colours['red'], colours['red'],
        colours['green'], colours['green'], colours['green'], colours['green_dark'],
        colours['purple'], colours['purple'], colours['purple'], colours['purple'], colours['purple'],
        colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'], colours['cyan'],
        colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'], colours['magenta'],
        colours['blue'], colours['blue'], colours['blue'], colours['blue'], colours['blue']]
    else:
        assert False, 'GetDefaultColours, unknown skeletonName'

    return jointColours

def PlotFrame(
    animation = None,
    parents = None,
    positions = None,
    frame = None,
    frame_info = None,
    skeleton:str = "Edinburgh", 
    axis_scale=5,
    elev=10,
    azim=180,
    dist=10,
    floorSize=20,
    floorSquares=50,
    display_grid:bool = False,
    ):

    if animation is not None:
        parents = animation.parents
        positions = Animation.positions_global(animation)
    elif (parents is None or positions is None ):
        raise AttributeError("Must provide either animation instance or both positions and parents")

    numJoints = len(parents)

    if animation is not None and frame is None:
        # draw T-pose if no frame number is provided
        positions = Animation.offsets_global(animation)
        title = "T-pose"
    elif frame is None:
        raise AttributeError("Must provide frame number if not providing Animation instance")
    else:
        positions = positions[frame]
        title = "Frame number: {:d}".format(frame)
    
    # deal with provided colour(s)
    if skeleton is not None:
        colours = np.array(GetDefaultColours(skeleton))/255.0
    else: 
        colours = ['red'] * numJoints

    # create figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-axis_scale, axis_scale)
    ax.set_zlim3d( -axis_scale, axis_scale)
    ax.set_ylim3d(0, axis_scale)
    ax.set_box_aspect((1,1,0.5), zoom=1)

    if display_grid:
        ax.grid(display_grid)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.set_axis_off()

    # initial camera position
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
    #ax.dist = dist

    # create floor
    xs = np.linspace(-floorSize, floorSize, floorSquares)
    zs = np.linspace(-floorSize, floorSize, floorSquares)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros(X.shape)
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.5)

    # display axes
    plt.plot([0,5], [0,0], [0,0], color='r', lw=4)
    plt.plot([0,0], [0,5], [0,0], color='g', lw=4)
    plt.plot([0,0], [0,0], [0,5], color='b', lw=4)
    

    plt.title(title)
    for j in range(1, numJoints): # start at 1 as we don't need line for root
        plt.plot([positions[j,0], positions[parents[j],0]],[positions[j,1], positions[parents[j],1]],[positions[j,2], positions[parents[j],2]], color=colours[j], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
    
    try:
        plt.show()
        #plt.save()
    except AttributeError as e:
        pass

    return

def PlotAnimation(
    animation = None,
    parents = None,
    positions = None,
    frame_info = None,
    skeleton:str = "Edinburgh", 
    filename=None,
    repeat=True,
    fps=60,
    axis_scale=5,
    elev=10,
    azim=180,
    dist=10,
    floorSize=20,
    floorSquares=50,
    display_grid:bool = False,
    ):
        
    """
    add functionality to prove a list of frame names - e.g. it mist have same shape as positions
        Parameters:
            animation: Animation to be played (type is Animation)
            colour: must be one of the following
                    - None, defaults to 'red'
                    - colour name as a string, e.g. 'red' or 'green' (if name not recognised by matplotlib an error will be returned)
                    - a single (r,g,b) tuple, either 0.0 < r,g,b < 1.0 or 0 < r,g,b < 255
                    - or else a list, with length equal to number of skeletal joints, of rgb-tuples (r,g,b)
            filename: if a final name is provided the animation will be saved in video format
            fps: frame rate at which to play/save animation (probably won't be able to play at this rate but saved video will be okay)

            elev:
            azim:
            dist:


    """

    # if animation is not None:
    #     parents = animation.parents
    #     positions = Animation.positions_global(animation)
    # elif (parents is None or positions is None ):
    #     raise AttributeError("Must provide either animation instance or both positions and parents")
    
    rootTranslation = positions[:,0,:]
    numFrames = positions.shape[0]
    numJoints = len(parents)

    if frame_info is not None and len(frame_info) == numFrames:
        display_titles = True
    else:
        display_titles = False

    # deal with provided colour(s)
    if skeleton is not None:
        colours = np.array(GetDefaultColours(skeleton))/255.0
    else: 
        colours = ['red'] * numJoints

    # create figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-axis_scale, axis_scale)
    ax.set_zlim3d( -axis_scale, axis_scale)
    ax.set_ylim3d(0, axis_scale)
    ax.set_box_aspect((1,1,0.5), zoom=1)
    
    if display_grid:
        ax.grid(display_grid)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.set_axis_off()

    # initial camera position
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
   # ax.dist = dist

    # create floor
    xs = np.linspace(-floorSize, floorSize, floorSquares)
    zs = np.linspace(-floorSize, floorSize, floorSquares)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros(X.shape)
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.5)

    # display axes
    plt.plot([0,5], [0,0], [0,0], color='r', lw=4)
    plt.plot([0,0], [0,5], [0,0], color='g', lw=4)
    plt.plot([0,0], [0,0], [0,5], color='b', lw=4)
    
    
    # these are the lines (of type 'mpl_toolkits.mplot3d.art3d.Line3D') that will be drawn. 
    # one for the root trajectory and then one for each bone
    # the first 2 or 3 arguments are lists of the x, y (, and z) coordinates of the points the line is to pass trye, 
    lines = []
    lines.append(plt.plot(rootTranslation[:,0], np.zeros(numFrames), rootTranslation[:,2], lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])[0])
    lines.append([plt.plot([0,0], [0,0], [0,0], color=colours[j], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for j in range(numJoints)])


    def animate(i):     

        if display_titles:
            ax.set_title("Frame: {:d}, {title}".format(i, title=frame_info[i]), y=0.95)
        else:
            ax.set_title("Frame: {:d}".format(i),  y=0.9) 
        for j in range(len(parents)):
            if parents[j] != -1:
                lines[1][j].set_data(np.array([[positions[i,j,0], positions[i,parents[j],0]],[positions[i,j,1],positions[i,parents[j],1]]]))
                lines[1][j].set_3d_properties(np.array([ positions[i,j,2],positions[i,parents[j],2]]))            
        return
        
    plt.tight_layout()
        
    ani = matplotlib.animation.FuncAnimation(fig,
        animate,
        np.arange(numFrames),
        interval=1000/fps,
        repeat=repeat)

    if filename != None:
        ani.save(filename, fps=fps, bitrate=13934)
        ani.event_source.stop()
        del ani
        plt.close()    
    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass

    return

def CompareFrames(
    animation1 = None,
    parents1 = None,
    positions1 = None,
    frame1 = None,
    frame_info1 = None,
    skeleton1:str = "Edinburgh", 
    animation2 = None,
    parents2 = None,
    positions2 = None,
    frame2 = None,
    frame_info2 = None,
    skeleton2:str = "Edinburgh", 
    axis_scale=5,
    elev=10,
    azim=180,
    dist=10,
    floorSize=20,
    floorSquares=50,
    display_grid:bool = False
    ):


    if animation1 is not None:
        parents1 = animation1.parents
        positions1 = Animation.positions_global(animation1)
    elif (parents1 is None or positions1 is None ):
        raise AttributeError("Must provide either animation instance or both positions and parents")

    if animation2 is not None:
        parents2 = animation2.parents
        positions2 = Animation.positions_global(animation2)
    elif (parents2 is None or positions2 is None ):
        raise AttributeError("Must provide either animation instance or both positions and parents")


    numJoints1 = len(parents1)
    numJoints2 = len(parents2)

    if animation1 is not None and frame1 is None:
        # draw T-pose if no frame number is provided
        positions1 = Animation.offsets_global(animation1)
        title1 = "T-pose"
    elif frame1 is None:
        raise AttributeError("Must provide frame number if not providing Animation instance")
    else:
        positions1 = positions1[frame1]
        title1 = "Frame number: {:d}".format(frame1)

    if frame2 is None:
        frame2 = frame1

    if animation2 is not None and frame2 is None:
        # draw T-pose if no frame number is provided
        positions2 = Animation.offsets_global(animation2)
        title2 = "T-pose"
    elif frame2 is None:
        raise AttributeError("Must provide frame number if not providing Animation instance")
    else:
        positions2 = positions2[frame2]
        title2 = "Frame number: {:d}".format(frame2)


    
    # deal with provided colour(s)
    if skeleton1 is not None:
        colours1 = np.array(GetDefaultColours(skeleton1))/255.0
    else: 
        colours1 = ['red'] * numJoints1
    if skeleton2 is not None:
        colours2 = np.array(GetDefaultColours(skeleton2))/255.0
    else: 
        colours2 = ['blue'] * numJoints2
    
    # create figure
    fig = plt.figure(figsize=(20,10))


    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim3d(-axis_scale, axis_scale)
    ax1.set_zlim3d( -axis_scale, axis_scale)
    ax1.set_ylim3d(0, axis_scale)
    ax1.set_box_aspect((1,1,0.5), zoom=1)
    if display_grid:
        ax1.grid(display_grid)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
    else:
        ax1.set_axis_off()

    # initial camera position
    ax1.view_init(elev=elev, azim=azim, vertical_axis='y')
    #ax1.dist = dist

    # create floor
    xs = np.linspace(-floorSize, floorSize, floorSquares)
    zs = np.linspace(-floorSize, floorSize, floorSquares)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros(X.shape)
    wframe1 = ax1.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.5)

    # display axes
    plt.plot([0,5], [0,0], [0,0], color='r', lw=4)
    plt.plot([0,0], [0,5], [0,0], color='g', lw=4)
    plt.plot([0,0], [0,0], [0,5], color='b', lw=4)

    plt.title(title1)
    for j in range(1, numJoints1): # start at 1 as we don't need line for root
        plt.plot([positions1[j,0], positions1[parents1[j],0]],[positions1[j,1], positions1[parents1[j],1]],[positions1[j,2], positions1[parents1[j],2]], color=colours1[j], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
 
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim3d(-axis_scale, axis_scale)
    ax2.set_zlim3d( -axis_scale, axis_scale)
    ax2.set_ylim3d(0, axis_scale)
    ax2.set_box_aspect((1,1,0.5), zoom=1)
    if display_grid:
        ax2.grid(display_grid)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
    else:
        ax2.set_axis_off()

    # initial camera position
    ax2.view_init(elev=elev, azim=azim, vertical_axis='y')
    #ax2.dist = dist

    # create floor
    wframe2 = ax2.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.5)

    # display axes
    plt.plot([0,5], [0,0], [0,0], color='r', lw=4)
    plt.plot([0,0], [0,5], [0,0], color='g', lw=4)
    plt.plot([0,0], [0,0], [0,5], color='b', lw=4)

    plt.title(title2)
    for j in range(1, numJoints2): # start at 1 as we don't need line for root
        plt.plot([positions2[j,0], positions2[parents2[j],0]],[positions2[j,1], positions2[parents2[j],1]],[positions2[j,2], positions2[parents2[j],2]], color=colours2[j], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])


    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass

    return

def CompareAnimations(
        animation1 = None,
        parents1 = None,
        positions1 = None,
        frame_info1 = None,
        skeleton1:str = "Edinburgh", 
        animation2 = None,
        parents2 = None,
        positions2 = None,
        frame_info2 = None,
        skeleton2:str = "Edinburgh", 
        filename=None,
        repeat=True,
        fps=60,
        axis_scale=5,
        elev1=10,
        azim1=180,
        dist1=10,
        elev2=10,
        azim2=180,
        dist2=10,
        floorSize=20,
        floorSquares=50,
        display_grid:bool = False
    ):
        
    """
        Case A) just one animation
        Case B) two animations and display them side by side

        Parameters:
            animation: either an Animation instance or a list of two Animation instances
            filename: if a final name is provided the animation will be saved in video format
    """

    if animation1 is not None:
        parents1 = animation1.parents
        positions1 = Animation.positions_global(animation1)
    elif (parents1 is None or positions1 is None ):
        raise AttributeError("Must provide either animation instance or both positions and parents")
    
    rootTranslation1 = positions1[:,0,:]
    numFrames1 = positions1.shape[0]
    numJoints1 = len(parents1)

    if frame_info1 is not None and len(frame_info1) == numFrames1:
        display_titles1 = True
    else:
        display_titles1 = False

    # deal with provided colour(s)
    if skeleton1 is not None:
        colours1 = np.array(GetDefaultColours(skeleton1))/255.0
    else: 
        colours1= ['red'] * numJoints1

    if animation2 is not None:
        parents2 = animation2.parents
        positions2 = Animation.positions_global(animation2)
    elif (parents2 is None or positions2 is None ):
        raise AttributeError("Must provide either animation instance or both positions and parents")
    
    rootTranslation2 = positions2[:,0,:]
    numFrames2 = positions2.shape[0]
    numJoints2 = len(parents2)

    if frame_info2 is not None and len(frame_info2) == numFrames2:
        display_titles2 = True
    else:
        display_titles2 = False

    # deal with provided colour(s)
    if skeleton2 is not None:
        colours2 = np.array(GetDefaultColours(skeleton2))/255.0
    else: 
        colours2= ['blue'] * numJoints2

    maxNumFrames = np.max([numFrames1, numFrames2])


    fig = plt.figure(figsize=(20,10)) # create figure

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim3d(-axis_scale, axis_scale)
    ax1.set_zlim3d( -axis_scale, axis_scale)
    ax1.set_ylim3d(0, axis_scale)
    ax1.set_box_aspect((1,1,0.5), zoom=1)
    if display_grid:
        ax1.grid(display_grid)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
    else:
        ax1.set_axis_off()


    ax1.view_init(elev=elev1, azim=azim1, vertical_axis='y')
    #ax1.dist = dist1

    # create floor
    xs = np.linspace(-floorSize, floorSize, floorSquares)
    zs = np.linspace(-floorSize, floorSize, floorSquares)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros(X.shape)
    wframe = ax1.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.5)

    # display axes
    plt.plot([0,5], [0,0], [0,0], color='r', lw=4)
    plt.plot([0,0], [0,5], [0,0], color='g', lw=4)
    plt.plot([0,0], [0,0], [0,5], color='b', lw=4)

    lines1 = []
    lines1.append(plt.plot(rootTranslation1[:,0],np.zeros(numFrames1),rootTranslation1[:,2], lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])[0])
    lines1.append([plt.plot([0,0], [0,0], [0,0], color=colours1[j], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for j in range(numJoints1)])



    # animation 
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim3d(-axis_scale, axis_scale)
    ax2.set_zlim3d( -axis_scale, axis_scale)
    ax2.set_ylim3d(0, axis_scale)
    ax2.set_box_aspect((1,1,0.5), zoom=1)
    if display_grid:
        ax2.grid(display_grid)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
    else:
        ax2.set_axis_off()

    ax2.view_init(elev=elev2, azim=azim2, vertical_axis='y')
   # ax2.dist = dist2
    
    wframe2 = ax2.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.5)
    # display axes
    plt.plot([0,5], [0,0], [0,0], color='r', lw=4)
    plt.plot([0,0], [0,5], [0,0], color='g', lw=4)
    plt.plot([0,0], [0,0], [0,5], color='b', lw=4)


    lines2 = []
    lines2.append(plt.plot(rootTranslation2[:,0],np.zeros(numFrames2),rootTranslation2[:,2], lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])[0])
    lines2.append([plt.plot([0,0], [0,0], [0,0], color=colours2[j], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for j in range(numJoints2)])

    def animate(i):
        
        if i < numFrames1:
            if display_titles1:
                ax1.set_title("Frame: {:d}, {title}".format(i, title=frame_info1[i]), y=0.95)
            else:
                ax1.set_title("Frame: {:d}".format(i),  y=0.9) 
            for j in range(len(parents1)):
                if parents1[j] != -1:
                    lines1[1][j].set_data(np.array([[positions1[i,j,0], positions1[i,parents1[j],0]],[positions1[i,j,1],positions1[i,parents1[j],1]]]))
                    lines1[1][j].set_3d_properties(np.array([ positions1[i,j,2],positions1[i,parents1[j],2]]))
     
        if i < numFrames2:
            if display_titles2:
                ax2.set_title("Frame: {:d}, {title}".format(i, title=frame_info2[i]), y=0.95)
            else:
                ax2.set_title("Frame: {:d}".format(i),  y=0.9) 
            for j in range(len(parents2)):
                if parents2[j] != -1:
                    lines2[1][j].set_data(np.array([[positions2[i,j,0], positions2[i,parents2[j],0]],[positions2[i,j,1],positions2[i,parents2[j],1]]]))
                    lines2[1][j].set_3d_properties(np.array([ positions2[i,j,2],positions2[i,parents2[j],2]]))

        return 
        
    plt.tight_layout()
        
    ani = matplotlib.animation.FuncAnimation(fig, 
        animate, np.arange(maxNumFrames), interval=1000/fps)

    if filename != None:
        ani.save(filename, fps=fps, bitrate=13934)
        ani.event_source.stop()
        del ani
        plt.close()    
    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass

    return

if __name__ == "__main__":
    print("Testing visualisation tools")

    from BVH import load
    from Animation import positions_global

    # PATH = "C:/Users/donal/source/data/EdinburghQuadruped/Raw/D1_008_KAN01_002.bvh"
    PATH = "C:/Users/donal/Downloads/lafan1/run2_subject1.bvh"
    PATH = "C:/Users/donal/source/data/PFNN/DogIK/LocomotionFlat05_000.bvh"
   
    anim, names, dt = load(PATH,  scale=0.01)

    titles = []
    titles2=[]
    for i in range(anim.shape[0]):
        titles.append(str(i)+str(i))
        titles2.append(str(i)+ "aaaa" + str(i))

    
    # PlotFrame(positions=positions_global(anim), parents = anim.parents,  skeleton="Edinburgh", frame=300 ) #
    # CompareFrames(positions1=positions_global(anim) ,parents1=anim.parents, positions2=positions_global(anim) ,parents2=anim.parents, frame1=100, frame2=800, skeleton1="Edinburgh", skeleton2="Edinburgh")
    # CompareAnimations(animation2=anim, positions1=positions_global(anim), parents1 = anim.parents, frame_info1=titles, frame_info2=[], skeleton1="Edinburgh", skeleton2="Edinburgh")
    PlotAnimation(anim, skeleton=None, frame_info=None, fps=30)