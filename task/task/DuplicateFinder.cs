using System.Collections.Generic;

namespace task
{
    // Easiest way to find duplicates though it's a prototype that doesn't work with csvs yet!
    public class DuplicateFinder
    {
        // Duplicate Data Finder
        public static string[] FindDuplicate(string[] data)
        {
            var duplicatedElements = new List<string>();
            // iterate over data and save them to a hashtable
            var dataHashSet = new HashSet<string>();

            foreach (var element in data)
            {
                if (dataHashSet.Contains(element)) duplicatedElements.Add(element);
                dataHashSet.Add(element);
            }

            return duplicatedElements.ToArray();
        }
    }
}