# Simulator Troubleshooting

There are several reasons why the simulator may be failing to start.
Here, we'll go over several checks which you can make to pinpoint the source of your issues.

Common issues:

- Missing binaries
- Binaries not compatible with system architecture
- Missing graphics drivers
- Missing display

## First Step: checking binaries

The most common cause of issues is missing or incomplete binary files.

To ensure that they exist, you can try to run the following commands
(several GBs of binary data will be downloaded!)

```
rm -rf ~/navdreams_binaries
git lfs clone git@github.com:ethz-asl/navrep3d_lfs.git ~/navdreams_binaries
```

Make sure to check that the binaries downloaded correctly, i.e.
` ~/navdreams_binaries/executables/... ` contains files,
and that the files in `~/navdreams_binaries/.gitattributes` were correctly downloaded by git-lfs.
The size of `~/navdreams_binaries` should be around 8 GBs.

## Second Step: simulator logs

When the unity binaries fail to run, the python client will usually display the following error:

```
ConnectionRefusedError: [Errno 111] Connection refused
```

A few lines above this error, the binary location is usually displayed, for example:

```
Set current directory to /home/ubuntu/navdreams_binaries/executables
Found path: /home/ubuntu/navdreams_binaries/executables/alternate.x86_64
```

To get more information on the cause of the error, run the binary with the logfile flag.

```
~/navdreams_binaries/executables/alternate.x86_64 -logfile log.txt
```

