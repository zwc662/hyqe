#! /bin/bash

rm -rf pyserini
git clone https://github.com/castorini/pyserini.git --branch pyserini-0.24.0 --recurse-submodules

if [ ! -d "./pyserini/tools/eval" ]; then
    read -p "Download submodule failed. Do you wish to clone from https://github.com/castorini/anserini-tools/tree/eac89401480fb340f96678487e018dcb7b4b5d66?" yn
    case $yn in
        [Yy]* ) cd ./pyserini/;
                git clone https://github.com/castorini/anserini-tools/;
                mv anserini-tools tools;
                cd tools;
                git checkout eac89401480fb340f96678487e018dcb7b4b5d66;
                cd ../../;
                break;;
        [Nn]* ) echo "Exit. Manual installation.";
                exit 1;
                break;;
    esac
fi

cd pyserini/tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../../
sudo apt-get update && apt-get install openjdk-11-jdk
wget https://repo1.maven.org/maven2/io/anserini/anserini/0.24.2/anserini-0.24.2-fatjar.jar
mv anserini-0.24.2-fatjar.jar ./pyserini/resources/jars/
python -m pip install -e . 

### May need to edit pyserini code when reporting error
file="./pyserini/search/_base.py"
for line_content in "'dl23': JTopics.TREC2023_DL," "'dl22-doc': JQrels.TREC2022_DL_DOC," "'dl23-doc': JQrels.TREC2023_DL_DOC," "'dl23-passage': JQrels.TREC2023_DL_PASSAGE,"; do
sed -i "s/${line_content}/# ${line_content}/" "$file"
done

cd ../
python -m pip install -e .
### Download a prebuilt index
echo "Do you wish to download contriever_msmarco_index?"
select yn in "Yes" "No"; do
    case $yn in
        "Yes" ) mkdir -p ./hyqe/hyqe/indexes; 
                cd ./hyqe/hyqe/indexes;
                wget  https://www.dropbox.com/s/dytqaqngaupp884/contriever_msmarco_index.tar.gz;
                tar -xvf contriever_msmarco_index.tar.gz; 
                cd ../../../
                break;;
        "No" ) break;;
    esac
done
