#	Makefile for czerny_wd.  Requires f2py from python-numpy be installed
#	If f2py fails, try using f2py --fcompiler=gnu95 -m myhelm_magstar -c myhelm_magstar.f90

all : myhelm_magstar.f90 const.dek helm_table_storage.dek implno.dek starmod_vector_eos.dek
	f2py -m myhelm_magstar -c myhelm_magstar.f90

clean:
	rm -rf *.so *.pyc
