# Running our code on runpod.io

This is a guide for running our code on [runpod.io](runpod.io).

**Note:** You pay both for running code and storing results afterwards if you don't terminate the pod.

## Deploy pod on runpod.io

1. Create user (and add funds)
1. Go to `Pods`
1. Choose GPU
1. Choose template `Runpod Pytorch 2.8.0`
1. Click `SSH Terminal Access` and `Start Jupyter Notebook`
1. Click `Deploy-on-demand`
1. Pres `Jupyter Lab` on runpod.io
1. Copy in the code your need (or use SSH+WinScp, see below for Windows users) 
1. Run `00_Instal.ipynb`
1. *Run whatever you like*

## SSH (optional)

1. In your terminal run (adjust user-path + e-mail):

    ```
    ssh-keygen -t ed25519 -f C:/Users/gmf123/.ssh/id_ed25519 -C “jeppe.druedahl@econ.ku.dk”
    ```

1. Open public key `C:/Users/gmf123/.ssh/id_ed25519.pub` and copy content
1. Add public key under `Settings` at ``SSH publich keys`` on runpod.io

## WinSCP (optional)

1. Find `SSH over exposed TCP` and copy SSH line: 

    ```
    ssh root@216.81.151.43-p 19979 -i ~/.ssh/id_ed25519
    ```

1. Connect in WinSCP with

    ```
    Host name: 216.81.151.43
    Port: 19979
    User: root
    Advanced -> SSH/Authentication -> Private key file = C:\Users\gmf123\.ssh\id_ed25519
    (it will convert it to .ppk format)
    ```

Transfer the files you need to `workspace` (not `root`).

## VSCode (alternative to JupyterLab)

1. Install extension: `SSH-Remote` 
1. Ctrl+Shift+P: `SSH-Remote Add New SSH Host` 
1. Paste SSH line copied above

    ```
    Host 216.81.151.43
    HostName 216.81.151.43
    User root
    Port 19979
    IdentityFile ~/.ssh/id_ed25519
    ```

1. Ctrl+Shift+P: `SSH-Remote: Connect to Host` 
1. Choose IP from SSH line
1. Choose `Linux`
1. Accept SSH fingerprint
1. Open folder