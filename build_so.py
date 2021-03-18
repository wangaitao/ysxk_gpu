# coding:utf-8
import sys, os, shutil, time
from distutils.core import setup
from Cython.Build import cythonize

starttime = time.time()
currdir = os.path.abspath('.')
parentpath = sys.argv[1] if len(sys.argv) > 1 else ""
setupfile = os.path.join(os.path.abspath('.'), __file__)
build_dir = "build"  # 项目加密后位置
build_tmp_dir = build_dir + "/temp"


def getpy(basepath=os.path.abspath('.'), parentpath='', name='', excepts=(), copyOther=False, delC=False):
    """
    获取py文件的路径
    :param basepath: 根路径
    :param parentpath: 父路径
    :param name: 文件/夹
    :param excepts: 排除文件（当前文件不是目标文件）
    :param copy: 是否copy其他文件
    :return: py文件的迭代器
    """
    # 返回所有文件夹绝对路径
    fullpath = os.path.join(basepath, parentpath, name)
    # 返回指定的文件夹包含的文件或文件夹的名字的列表
    for fname in os.listdir(fullpath):
        ffile = os.path.join(fullpath, fname)
        # print('fname',fname)
        # print("ffile", ffile)
        # print basepath, parentpath, name,file
        # 是文件夹 且不以.开头 不是 build  ，不是迁移文件
        if os.path.isdir(ffile) and fname != build_dir and not fname.startswith('.') and fname != "migrations":
            # 循环遍历文件夹
            for f in getpy(basepath, os.path.join(parentpath, name), fname, excepts, copyOther, delC):
                yield f
        elif os.path.isfile(ffile):
            # 筛选出 .c 文件
            ext = os.path.splitext(fname)[1]
            if ext == ".c":
                # 显示文件 "ffile" 信息,st_mtime: 最后一次修改的时间。
                if delC and os.stat(ffile).st_mtime > starttime:
                    # 删除 .c 文件
                    os.remove(ffile)
            # 文件不是排除文件  且不是'.pyc' '.pyx'文件
            elif ffile not in excepts and os.path.splitext(fname)[1] not in ('.pyc', '.pyx'):
                # manage.py文件不编译
                if os.path.splitext(fname)[1] in ('.py', '.pyx') and not fname.startswith(
                        '__') and fname != "manage.py" and fname != "test.py":
                    # 返回要加密的文件（加到module_list中成为一个列表）
                    yield os.path.join(parentpath, name, fname)
                elif copyOther:
                    dstdir = os.path.join(basepath, build_dir, parentpath, name)
                    # 判断文件夹是否存在，不存在则创建
                    if not os.path.isdir(dstdir):
                        os.makedirs(dstdir)
                        print('dstdir的是',dstdir)
                    # 复制文件到新文件夹下
                    print('ffile的是',ffile)
                    shutil.copyfile(ffile, os.path.join(dstdir, fname))
        else:
            pass

# 获取py列表
module_list = list(getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile)))
print(module_list)
try:
    setup(ext_modules=cythonize(module_list), script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir])
except Exception as e:
    print(e)
else:
    module_list = list(getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile), copyOther=True))
module_list = list(getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile), delC=True))
# shutil.rmtree() 表示递归删除文件夹下的所有子文件夹和子文件
# 删除build_tmp_dir临时文件夹
if os.path.exists(build_tmp_dir):
    shutil.rmtree(build_tmp_dir)
print("complate! time:", time.time() - starttime, 's')
