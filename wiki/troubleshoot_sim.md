# Simulator Troubleshooting

There are several reasons why the simulator may be failing to start.
Here, we'll go over several checks which you can make to pinpoint the source of your issues.

Common issues:

- Missing binaries
- Binaries not compatible with system architecture
- Missing graphics drivers
- Missing display
- Failing to connect to socket at 127.0.0.1:25001 (for example in Virtual Machine)

## First Step: Checking Binaries

The most common cause of issues is missing or incomplete binary files.

To ensure that they exist, you can try to run the following commands
(several GBs of binary data will be downloaded!)

```
cd ~
wget http://robotics.ethz.ch/~asl-datasets/2022_NavDreams/navdreams_binaries_v1.3.tar
tar -xvf navdreams_binaries_v1.3.tar
```

Make sure to check that the binaries downloaded correctly, i.e.
` ~/navdreams_binaries/executables/... ` contains files,
and that the files in `~/navdreams_binaries/.gitattributes` were correctly downloaded.
The size of `~/navdreams_binaries` should be around 8 GBs.



## Second Step: Unity Logs

When the unity binaries fail to run, the python client will usually display the following error:

```
ConnectionRefusedError: [Errno 111] Connection refused
```

A few lines above this error, the binary location is usually displayed, for example:

```
Set current directory to /home/ubuntu/navdreams_binaries/mlagents_executables
Found path: /home/ubuntu/navdreams_binaries/mlagents_executables/cathedral.x86_64
```

There can be several reasons for the python client failing to communicate with the unity simulator.
To get more information on the cause of the error, run the binary with the logfile flag.

```
~/navdreams_binaries/mlagents_executables/cathedral.x86_64 -logfile log.txt
```

If a display is found, the logfile should contain lines similar to:

```
Display 0 'NVIDIA VGX  32"': 1024x768 (primary device).
Desktop is 1024 x 768 @ 170 Hz
```

However, if you see

```
Desktop is 0 x 0 @ 0 Hz
```

This could indicate that a display is not set up properly. Make sure that the command `echo $DISPLAY` returns a value like `:0`. If not, refer to [this post](https://dugas.ch/lord_of_the_files/run_your_unity_ml_executable_in_the_cloud.html) for help.

Similarly, if there are error messages about missing OpenGL libraries, make sure that the command 

```
DISPLAY=:0 glxinfo
```

returns a valid OpenGL configuration, with high enough version for unity.  If not, refer to [this post](https://dugas.ch/lord_of_the_files/run_your_unity_ml_executable_in_the_cloud.html) for help.

## The Simulator Is Running, But Output Is Strange

If the visuals are strange (missing textures, etc), most probably parts of the binaries are missing (red error messages should show up in the unity simulator window). Make sure to download them as detailed in the above section.

Rarely, we observed that on some EC2 instances specific binaries run strangely (very slowly). We suspect a driver incompatibility. This brings up to the last resort solution to binaries which refuse to run:

## Build Binaries From Source

If nothing else works, there is always the possibility of building the binaries from the source Unity project on the target architecture.

Refer to [NavDreamsUnity](https://www.github.com/ethz-asl/NavDreamsUnity) for more information.


