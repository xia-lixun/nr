# dnn
deep neural network

1. nn example based on mnist data




## Git
* Undo a commit and redo  
>$ git commit -m "Something terribly misguided"              
>$ git reset HEAD~                                           
><< edit files as necessary >>                               
>$ git add ...                                               
>$ git commit [-a] -c ORIG_HEAD                              


* git pull remote branch to local: need to make local track  
>$ git checkout --track origin/dev1222

* undo uncommited files  
>$ git reset --hard  
>$ git clean -fd

* merge branches
>$ git checkout master  
>$ git merge dev1222  
>$ git branch -d dev1222  only if you delete the branch  



## SMBA
> sudo apt install samba  
> sudo gedit /etc/samba/smb.conf &  

> [share]  
>comment = Ubuntu File Server Share  
>path = /path/to/the/folder  
>browsable = yes  
>guest ok = yes  
>read only = no  
>writeable = yes  

> sudo chmod -R 777 /path/to/the/folder  
> sudo service smbd restart  
> sudo service nmbd restart  

Don't forget to install ssh server:
>sudo apt install openssh-server


## ZFS
1. sudo apt install zfsutils-linux
2. sudo zfs list
3. df -h
4. sudo fdisk -l
5. sudo zpool create -f oak /dev/sdc /dev/sdd /dev/sde  (raid-0, adaptive stripe)
6. sudo zfs list
7. sudo zpool status
8. sudo zpool create -f oak raidz /dev/sdc /dev/sdd /dev/sde
9. sudo zpool add -f oak raidz /dev/sdf /dev/sdg /dev/sdh
