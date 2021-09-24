# Getting started
This part contains

## Accessing Alvis
Now that you've [gotten access](https://www.c3se.chalmers.se/documentation/getting_access/), are [getting started](https://www.c3se.chalmers.se/documentation/getting_started/) and have attended the [introduction presentation for Alvis](https://www.c3se.chalmers.se/documentation/intro-alvis/slides/) you are certainly itching to access Alvis and start doing stuff.

To access Alvis there are a few different alternatives and they can all be found at [c3se.chalmers.se](c3se.chalmers.se):
 - [Connecting through terminal](https://www.c3se.chalmers.se/documentation/connecting/)
 - [Remote graphics](https://www.c3se.chalmers.se/documentation/remote_graphics/)
 - [Remote development with Visual Studio Code](https://www.c3se.chalmers.se/documentation/remote-vscode/remote_vscode/)

## Using Alvis
In this part we will explore how to get started with using Alvis as a first time user. As a first step you should connect to Alvis (see previous section) and now you should have access to Alvis through a terminal or if you are using Thinlinc or VS Code you can open a terminal.

If there is any command that you are unsure of what it does you can use the command `man` e.g to find out about ls
```bash
man ls
```
or
```bash
ls --help
```

If you want to abort a command pressing `ctrl + c` is usually the way to go.

### Get tutorial files
Now in this terminal we want to get this tutorial either clone the repository
```bash
[CID@alvis1 ~]$ git clone https://github.com/c3se/alvis-intro.git
```
or download it as an archive
```bash
[CID@alvis1 ~]$ wget https://github.com/c3se/alvis-intro/archive/refs/heads/main.zip
[CID@alvis1 ~]$ mv alvis-intro-main alvis-intro
```

Now lets move to this file
```bash
[CID@alvis1 ~]$ cd alvis-intro/tutorial/part1
[CID@alvis1 part1]$ ls
hello.sh  README.md
```

To read this file you can use your favourite command line text editor, `cat` or `less`
```bash
[CID@alvis1 part1]$ less README.md
```
to go out from `less` press `q`.

### Submitting a file with srun
Before you submit a file you want to make sure that there are no obvious errors before you submit it, to do this it is absolutely OK to run small tests directly on the log-in node.

Take a look at the contents of `hello.sh` and if you think that it looks ok you can try running it on the log-in node.
