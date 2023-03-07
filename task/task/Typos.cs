using System;

namespace task
{
    public class Typos
    {
        public static bool IsTypo(string word)
        {
            string[] dictionary = { "apple", "banana", "cherry", "orange", "pear", "pineapple", "strawberry" };
            foreach (var dictWord in dictionary)
            {
                var distance = LevenshteinDistance(word, dictWord);
                if (distance <= 2) return true;
            }

            return false;
        }

        public static int LevenshteinDistance(string s1, string s2)
        {
            int m = s1.Length, n = s2.Length;
            var dp = new int[m + 1, n + 1];
            for (var i = 0; i <= m; i++) dp[i, 0] = i;
            for (var j = 0; j <= n; j++) dp[0, j] = j;
            for (var i = 1; i <= m; i++)
            for (var j = 1; j <= n; j++)
                if (s1[i - 1] == s2[j - 1])
                    dp[i, j] = dp[i - 1, j - 1];
                else
                    dp[i, j] = 1 + Math.Min(Math.Min(dp[i - 1, j], dp[i, j - 1]), dp[i - 1, j - 1]);
            return dp[m, n];
        }
    }
}
// READ ME //
// This is the way to run the code
/*
// "skill" "child" are not a typo
// "banan" "applle" are typos
string word = "chery";
bool isTypo = Typos.IsTypo(word);
if (isTypo)
{
    Console.WriteLine("The word \"" + word + "\" is a potential typo.");
}
else
{
    Console.WriteLine("The word \"" + word + "\" is not a typo.");
}
*/