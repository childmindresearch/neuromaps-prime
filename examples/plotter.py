from neuromaps.datasets import fetch_fslr

import yaspy

surfaces = fetch_fslr()
surf_path, _ = surfaces["inflated"]
sulc_path, _ = surfaces["sulc"]

plotter = yaspy.Plotter(surf_path, hemi="lh", sulc=sulc_path)
plotter.screenshot(view="lateral")