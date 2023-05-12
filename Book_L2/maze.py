# 2023/05/12 kerong


#  plugin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import display

# define graph size ans name
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()

# draw red wall
plt.plot([1, 1], [0, 1], color="red", linewidth=2)
plt.plot([1, 2], [2, 2], color="red", linewidth=2)
plt.plot([2, 2], [2, 1], color="red", linewidth=2)
plt.plot([2, 3], [1, 1], color="red", linewidth=2)

# draw number
plt.text(0.5, 2.5, "S0", size=14, ha="center")
plt.text(1.5, 2.5, "S1", size=14, ha="center")
plt.text(2.5, 2.5, "S2", size=14, ha="center")
plt.text(0.5, 1.5, "S3", size=14, ha="center")
plt.text(1.5, 1.5, "S4", size=14, ha="center")
plt.text(2.5, 1.5, "S5", size=14, ha="center")
plt.text(0.5, 0.5, "S6", size=14, ha="center")
plt.text(1.5, 0.5, "S7", size=14, ha="center")
plt.text(2.5, 0.5, "S8", size=14, ha="center")
plt.text(0.5, 2.3, "START", ha="center")
plt.text(2.5, 0.3, "GOAL", ha="center")

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

plt.tick_params(axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="off",
                right="off",
                left="off",
                labelleft="off")

line, = ax.plot([0.5], [2.5], marker="o", color="g", markersize=60)

# up right down left
theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                    [np.nan, 1, np.nan, 1],  # s1
                    [np.nan, np.nan, 1, 1],  # s2
                    [1, 1, 1, np.nan],  # s3
                    [np.nan, np.nan, 1, 1],  # s4
                    [1, np.nan, np.nan, np.nan],  # s5
                    [1, np.nan, np.nan, np.nan],  # s6
                    [1, 1, np.nan, np.nan],  # s7
                    ])


# custom parameter theta covert to pi function
def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape  # get theta size
    pi = np.zeros((m, n))

    for i in range(m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # calculate rate

    pi = np.nan_to_num(pi)  # set nan to 0

    return pi


pi_0 = simple_convert_into_pi_from_theta(theta_0)


# custom calculate after 1 step state

def get_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    s_next = -1

    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        s_next = s - 3
    elif next_direction == "right":
        s_next = s + 1
    elif next_direction == "down":
        s_next = s + 3
    elif next_direction == "left":
        s_next = s - 1

    return s_next


# In[8]:


def goal_maze(pi):
    global state_history
    s = 0
    state_history = [0]

    while True:
        next_s = get_next_s(pi, s)
        state_history.append(next_s)

        if next_s == 8:
            break
        else:
            s = next_s
    return state_history


state_history = goal_maze(pi_0)
print(state_history)
print("走了" + str(len(state_history) - 1) + "步")


def init():
    line.set_data([], [])
    return line,


def animate(i):
    state = state_history[i]
    x = [(state % 3) + 0.5]
    y = [2.5 - int(state / 3)]
    line.set_data(x, y)
    return line,


anim = animation.FuncAnimation(fig,
                               animate,
                               init_func=init,
                               frames=len(state_history),
                               interval=200,
                               repeat=False)

# plt.show()

with open("My_html.html", "w") as f:
    print(anim.to_jshtml(), file=f)

