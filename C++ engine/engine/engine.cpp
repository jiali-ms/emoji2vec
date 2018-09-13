// engine.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "w2v.h"

#include <Windows.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <codecvt>
#include <locale>

using namespace w2v;

int main()
{
    auto model = W2VModel();

    std::ofstream utf8file("test.txt");
    std::wbuffer_convert<std::codecvt_utf8<wchar_t>> converter(utf8file.rdbuf());
    std::wostream out(&converter);

    auto score = model.Similarity(L"king", L"queen");
    out << L"similarity between king and queen is " << score << std::endl;

    auto v = model.ContextVector({ L"beijing", L"seoul" });
    auto sim = model.Similarity(v, model.WordVector(L"tokyo"));
    out << L"similarity between beijing + osaka and tokyo is " << sim << std::endl;

    auto test_word = L"happy";
    auto similar = model.MostSimilar(test_word, 100);

    out << L"top similar words to happy are:  " << std::endl;
    for (auto word : similar)
    {
        out << word << " ";
    }

    out << L"top similar emoji to happy are:  " << std::endl;
    for (auto word : similar)
    {
        if (model.IsEmoji(word))
        {
            out << word << " ";
        }
    }

    utf8file.close();

    return 0;
}

