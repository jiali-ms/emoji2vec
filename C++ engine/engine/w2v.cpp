#include "stdafx.h"
#include "w2v.h"

namespace w2v {

    W2VModel::W2VModel()
    {
        LoadVocab(s_rootPath + "vocab.txt");
        LoadCode(s_rootPath + "code.txt");
        LoadCodebook(s_rootPath + "codebook.txt");
    }

    int W2VModel::Encode(const std::wstring & s)
    {
        return _w2i.count(s) < 1 ? -1 : _w2i[s];
    }

    std::vector<float> W2VModel::WordVector(const std::wstring & word)
    {
        if (_w2i.count(word) < 1)
        {
            return std::vector<float>();
        }

        return VectorAt(_w2i[word]);
    }

    std::vector<std::wstring> W2VModel::MostSimilar(const std::wstring &target, int topn)
    {
        auto ret = std::vector<std::wstring>();
        auto v = std::vector<std::pair<double, int>>();

        if (_w2i.count(target) < 1)
        {
            return ret;
        }

        // calculate cosin distance scores with others
        auto idx = _w2i[target];
        auto t = L2Norm(VectorAt(idx));
        for (int i = 0; i < s_vocabSize; i++)
        {
            if (i != idx)
            {
                auto s = L2Norm(VectorAt(i));
                v.push_back(std::pair<double, int>(CosineDistance(t, s), i));
            }
        }

        // sort by consine distance score
        std::sort(v.begin(), v.end(), [](const std::pair<double, int> &left, const std::pair<double, int> &right) {
            return left.first > right.first;
        });

        for (int i = 0; i < topn; i++)
        {
            ret.push_back(_i2w[v[i].second]);
        }

        return ret;
    }

    float W2VModel::Similarity(const std::wstring &s1, const std::wstring &s2)
    {
        if (_w2i.count(s1) < 1 || _w2i.count(s2) < 1)
        {
            return 0.0;
        }

        auto idx1 = _w2i[s1];
        auto idx2 = _w2i[s2];

        return Similarity(idx1, idx2);
    }

    float W2VModel::Similarity(int idx1, int idx2)
    {
        if (idx1 < 0 || idx2 < 0)
        {
            return 0;
        }

        return CosineDistance(L2Norm(VectorAt(idx1)), L2Norm(VectorAt(idx2)));
    }

    float W2VModel::Similarity(const std::vector<float>& v1, const std::vector<float>& v2)
    {
        return CosineDistance(L2Norm(v1), L2Norm(v2));
    }

    std::vector<float> W2VModel::ContextVector(const std::vector<std::wstring> &contextWords)
    {
        auto ret = std::vector<float>();
        auto count = 0;
        for (auto word : contextWords)
        {
            if (_w2i.count(word) > 0)
            {
                auto v = VectorAt(_w2i[word]);

                for (size_t i = 0; i < v.size(); i++)
                {
                    if (ret.size() < v.size())
                    {
                        ret.push_back(v[static_cast<int>(i)]);
                    }
                    else
                    {
                        ret[i] += v[i];
                    }
                }

                count += 1;
            }
        }

        for (size_t i = 0; i < ret.size(); i++)
        {
            ret[static_cast<int>(i)] /= count;
        }

        return ret;
    }

    bool W2VModel::IsEmoji(const std::wstring & emoji)
    {
        if (_w2i.count(emoji) > 0)
        {
            return _emoji_idx.count(_w2i[emoji]) > 0;
        }

        return false;
    }

    std::vector<float> W2VModel::VectorAt(int i)
    {
        auto ret = std::vector<float>();
        for (size_t j = 0; j < s_embeddingSize; j++)
        {
            ret.push_back(_codebook[_code[i][j]]);
        }

        return ret;
    }

    float W2VModel::CosineDistance(const std::vector<float>& v1, const std::vector<float>& v2)
    {
        float ret = 0.0f;
        for (size_t i = 0; i < v1.size(); i++)
        {
            ret += v1[i] * v2[i];
        }

        return ret;
    }

    std::vector<float> W2VModel::L2Norm(const std::vector<float>& v)
    {
        auto ret = std::vector<float>();
        double accum = 0.0;
        for (size_t i = 0; i < v.size(); i++)
        {
            accum += v[i] * v[i];
        }
        accum = static_cast<float>(std::sqrt(accum));

        for (size_t i = 0; i < v.size(); i++)
        {
            ret.push_back(static_cast<float>(v[i] / accum));
        }

        return ret;
    }

    void W2VModel::LoadVocab(const std::string & path)
    {
        std::wifstream  fin(path);
        fin.imbue(std::locale(std::locale::empty(), new std::codecvt_utf8<wchar_t>));

        if (fin.is_open())
        {
            for (int i = 0; i < s_vocabSize; i++)
            {
                std::wstring item;
                fin >> item;
                _w2i[item] = i;
                _i2w[i] = item;
                int is_emoji;
                fin >> is_emoji;
                if (is_emoji)
                {
                    _emoji_idx.insert(i);
                }
            }

            fin.close();
        }
    }

    void W2VModel::LoadCode(const std::string & path)
    {
        std::ifstream fin(path);

        if (fin.is_open())
        {
            for (int i = 0; i < s_vocabSize; i++)
            {
                auto row = std::vector<unsigned char>();
                for (int j = 0; j < s_embeddingSize; j++)
                {
                    int item = 0;
                    fin >> item;
                    row.push_back(static_cast<unsigned char>(item));
                }
                _code.push_back(row);
            }

            fin.close();
        }
    }

    void W2VModel::LoadCodebook(const std::string & path)
    {
        std::ifstream fin(path);

        if (fin.is_open())
        {
            for (int i = 0; i < s_codebookSize; i++)
            {
                float item = 0.0;
                fin >> item;
                _codebook.push_back(item);
            }

            fin.close();
        }
    }
}