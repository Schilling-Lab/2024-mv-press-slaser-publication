# hypermri - Testing 🧲🕵️

## Test Data
The data for both testing and example notebooks is stored inside a seperate repository. To download it please follow the steps below:

### 1. Download Test Data
The dataset for all tests is managed inside an own repository. To download the test data simply clone the repository.

Inside this '*tests*' directory run:

```bash
$ git clone https://gitlab.lrz.de/mri_nuk/python/bruker-data-analysis-test-data.git data
```

### 2. Install Git LSF
For large binary files it is essential to use [**Git LSF**](https://git-lfs.com). Please go to the [homepage](https://git-lfs.com) and download and install it.

After the installation process make sure everything works by running the following inside the test data repository:
```bash
$ git lfs install
```

#### Files We Track:
The following files are already tracked:
- *fid*
- *2dseq*
- *rawdata.job[...]*

#### Track New Files:
In case you have a file that is not mentioned in the chapter *Files We Track*, **follow the steps below before adding the new file type to the repository.**
1. To track new files, run `git lfs track`. You can test your matching pattern [here](https://www.digitalocean.com/community/tools/glob). Two examples are shown below:
    ```bash
    $ git lfs track "**/fid"
    $ git lfs track "**/rawdata.job[0-9]*"
    ```
2. Commit all changes the `git lfs track` command left in *.gitattributes*:  
    ```bash
    $ git add .gitattributes
    $ git commit -m "Added file type ... to .gitattributes to capture LFS tracking"
    ```
3. Add the new file type to this README in section *Files We Track*.

4. Now you are ready to include the new file type in your commits.
## Run Tests

Testing is done using [**pytest**](https://docs.pytest.org/en/7.2.x/). Inside the hypermri directory run:

```bash
$ pytest
```
