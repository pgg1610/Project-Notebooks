import matplotlib.pyplot as plt
import numpy as np

def illustrate_1d_gpr(f, gpr, x_plot, EI, LCB):
    x_sample = gpr.X_train_
    y_sample = gpr.y_train_
    (mean, stdev) = gpr.predict(x_plot, return_std = True)
    mean = mean.flatten()
    
    fig = plt.figure(figsize = (8, 9))
    ax1 = fig.add_subplot(311)
    ax1.plot(x_plot, f(x_plot), "k-", label = "Ground Truth", zorder = 0)
    ax1.plot(x_plot, mean, "r--", label = "Predicted Mean", zorder = 1)
    ax1.plot(x_sample, y_sample, "ro", label = "Sampled Points", zorder = 2)
    ax1.fill_between(x_plot.flatten(), mean - 1.96*stdev, mean + 1.96*stdev, color = "r", alpha = 0.25)
    ax1.set_xlabel(r"$x$", fontsize = 15)
    ax1.set_ylabel(r"$f(x)$", fontsize = 15)
    ax1.set_title("Gaussian Process Regression", fontsize = 15)
    ax1.legend(fontsize = 10)
    ax1.grid(True)
    
    ax2 = fig.add_subplot(312)
    ax2.plot(x_plot, EI(x_plot, gpr), "k-")
    ax2.set_xlabel(r"$x$", fontsize = 15)
    ax2.set_ylabel(r"$EI(x)$", fontsize = 15)
    ax2.set_title(r"Expected Improvement, $\delta=%.2f$" % EI.params["delta"], fontsize = 15)
    ax2.grid(True)
    
    ax3 = fig.add_subplot(313)
    ax3.plot(x_plot, LCB(x_plot, gpr), "k-")
    ax3.set_xlabel(r"$x$", fontsize = 15)
    ax3.set_ylabel(r"$LCB(x)$", fontsize = 15)
    ax3.set_title(r"Lower Confidence Bound, $\sigma=%.2f$" % LCB.params["sigma"], fontsize = 15)
    ax3.grid(True)
    plt.tight_layout()
    
    return (fig, (ax1, ax2, ax3))

def illustrate_2d_gpr(f, gpr, x_plot, y_plot, LCB):
    x_sample = gpr.X_train_
    xy_plot = np.asarray([z.flatten() for z in np.meshgrid(x_plot, y_plot)]).T
    f_plot = f(xy_plot).reshape((y_plot.size, x_plot.size)) # objective function
    pm_plot = gpr.predict(xy_plot).reshape((y_plot.size, x_plot.size)) # predicted mean
    lcb_plot = LCB(xy_plot, gpr).reshape((y_plot.size, x_plot.size)) # lower confidence bound
    
    global_min = min(f_plot.min(), lcb_plot.min())
    global_max = max(f_plot.max(), f_plot.max())
    
    fig = plt.figure(figsize = (13, 4))
    ax1 = fig.add_subplot(131)
    ax1.contourf(x_plot, y_plot, f_plot, cmap = "Reds", vmin = global_min, vmax = global_max)
    ax1.plot(x_sample[:,0], x_sample[:,1], "ko")
    ax1.set_xlabel(r"$x_1$", fontsize = 15)
    ax1.set_ylabel(r"$x_2$", fontsize = 15)
    ax1.set_title(r"Objective, $f(x_1,x_2)$", fontsize = 15)
    
    ax2 = fig.add_subplot(132)
    ax2.contourf(x_plot, y_plot, pm_plot, cmap = "Reds", vmin = global_min, vmax = global_max)
    ax2.plot(x_sample[:,0], x_sample[:,1], "ko")
    ax2.set_xlabel(r"$x_1$", fontsize = 15)
    ax2.set_ylabel(r"$x_2$", fontsize = 15)
    ax2.set_title(r"Predicted Mean, $\hat{\mu}(x_1,x_2)$", fontsize = 15)
    
    ax3 = fig.add_subplot(133)
    cs = ax3.contourf(x_plot, y_plot, lcb_plot, cmap = "Reds", vmin = global_min, vmax = global_max)
    ax3.plot(x_sample[:,0], x_sample[:,1], "ko")
    ax3.set_xlabel(r"$x_1$", fontsize = 15)
    ax3.set_ylabel(r"$x_2$", fontsize = 15)
    ax3.set_title(r"$LCB(x_1,x_2;\sigma=%.2f)$" % LCB.params["sigma"], fontsize = 15)
    
    plt.tight_layout()
    fig.subplots_adjust(right = 0.93)
    cbar_ax = fig.add_axes([0.95, 0.16, 0.04, 0.72])
    fig.colorbar(cs, cax = cbar_ax)
    
    return (fig, (ax1, ax2, ax3))

def main(*args, **kwargs):
    pass

if __name__ == "__main__":
    main()