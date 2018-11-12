mkdir modelzoo

cd modelzoo

git clone https://github.com/dhairyagandhi96/model-zoo.git

git checkout master

cd model-zoo/script
~/julia-1.0.0/bin/julia notebook.jl

cd ../..

rm -rf modelzoo
