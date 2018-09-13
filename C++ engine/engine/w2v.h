#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <codecvt>
#include <set>
#include <locale>

// pre-defined from pipeline
static const int s_vocabSize = 23070;
static const int s_embeddingSize = 100;
static const int s_codebookSize = 16;
static const std::string s_rootPath = "E:/emoji2vec/data/comp_4/";

namespace w2v {

    class W2VModel
    {
    public:
        W2VModel();

        // find most similar words of the target word
        // Note that this is a full vocab search
        std::vector<std::wstring> MostSimilar(const std::wstring &target, int topn);

        int Encode(const std::wstring &s);

        std::vector<float> WordVector(const std::wstring &word);

        // For OOV word, just return 0 as no relationship
        float Similarity(const std::wstring &s1, const std::wstring &s2);

        // For OOV word, just return 0 as no relationship
        float Similarity(int idx1, int idx2);

        float Similarity(const std::vector<float> &v1, const std::vector<float> &v2);

        std::vector<float> ContextVector(const std::vector<std::wstring> &contextWords);

        bool IsEmoji(const std::wstring &emoji);

    private:
        void LoadVocab(const std::string &path);
        void LoadCode(const std::string &path);
        void LoadCodebook(const std::string &path);

        std::vector<float> VectorAt(int i);
        float CosineDistance(const std::vector<float> &v1, const std::vector<float> &v2);
        std::vector<float> L2Norm(const std::vector<float> &v);

        std::map<std::wstring, int> _w2i;
        std::map<int, std::wstring> _i2w;
        std::vector<std::vector<unsigned char>> _code;
        std::vector<float> _codebook;
        std::set<int> _emoji_idx;
    };

}