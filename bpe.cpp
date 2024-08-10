#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <stdexcept>
#include <fstream>
#include <sstream>

const int MAX_VOCAB_SIZE = 1000;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

std::vector<int> string_to_byte(const std::string& input, const std::string& encoding) {
    std::vector<int> result;
    for (unsigned char c : input) {
        result.push_back(static_cast<int>(c));
    }
    return result;
}

std::pair<std::pair<int, int>, int> most_frequent_pair(const std::vector<int>& indices) {
    std::unordered_map<std::pair<int, int>, int, pair_hash> counts;
    for (size_t i = 0; i < indices.size() - 1; ++i) {
        counts[{indices[i], indices[i + 1]}]++;
    }
    
    if (counts.empty()) {
        return {{-1, -1}, 0};
    }
    
    auto max_element = std::max_element(
        counts.begin(), counts.end(),
        [](const auto& p1, const auto& p2) { return p1.second < p2.second; }
    );
    
    return {max_element->first, max_element->second};
}

std::vector<int> merge_pair(const std::vector<int>& indices, const std::pair<int, int>& pair, int new_index) {
    std::vector<int> merged;
    size_t i = 0;

    while (i < indices.size()) {
        if (i < indices.size() - 1 && std::make_pair(indices[i], indices[i + 1]) == pair) {
            merged.push_back(new_index);
            i += 2;
        } else {
            merged.push_back(indices[i]);
            i++;
        }
    }

    return merged;
}

class BPETokenizer {
public:
    BPETokenizer(int max_vocab_size) : max_vocab_size(max_vocab_size) {
        if (max_vocab_size <= 256) {
            throw std::invalid_argument("Maximum vocabulary size must be at least 256");
        }
        reset();
    }

    void reset() {
        pairs.clear();
        id_to_token.clear();
        for (int i = 0; i < 256; ++i) {
            id_to_token[i] = std::string(1, static_cast<char>(i));
        }
        next_id = 256;
        special_to_id.clear();
        id_to_special.clear();
    }

    void register_special_token(const std::string& token) {
        if (special_to_id.find(token) == special_to_id.end()) {
            std::cout << "Added special token " << token << " with ID " << next_id << "\n" << std::endl;
            special_to_id[token] = next_id;
            id_to_special[next_id] = token;
            next_id++;
        }
    }

    void train(const std::string& input, bool stop_early = false, bool verbose = false) {
        std::vector<int> indices = string_to_byte(input, "utf-8");

        while (vocab_size() < max_vocab_size) {
            auto [pair, count] = most_frequent_pair(indices);

            if (pair.first == -1 && pair.second == -1) {
                break;
            }

            if (stop_early && count == 1) {
                break;
            }

            indices = merge_pair(indices, pair, next_id);
            std::string new_token = id_to_token[pair.first] + id_to_token[pair.second];
            pairs[pair] = next_id;
            id_to_token[next_id] = new_token;

            if (verbose) {
                std::cout << "Merged IDs (" << pair.first << ", " << pair.second << ") as a new token \""
                        << new_token << "\" with ID " << next_id << "\n" << std::endl;
            }

            next_id++;
        }

        std::cout << "Training complete: " << pairs.size() << " merges performed. Final vocabulary size: " 
                  << vocab_size() << "\n" << std::endl;
    }

    std::vector<int> encode(const std::string& input) {
        std::vector<int> indices;
        
        if (!special_to_id.empty()) {
            std::string pattern = "(";
            for (const auto& [token, _] : special_to_id) {
                if (pattern.length() > 1) pattern += "|";
                pattern += std::regex_replace(token, std::regex("[.^$*+?()[\\]{}|]"), "\\$&");
            }
            pattern += ")";
            std::regex special_pattern(pattern);
            
            std::sregex_token_iterator iter(input.begin(), input.end(), special_pattern, {-1, 0});
            std::sregex_token_iterator end;
            
            for (; iter != end; ++iter) {
                std::string split = *iter;
                if (special_to_id.find(split) != special_to_id.end()) {
                    indices.push_back(special_to_id[split]);
                } else {
                    auto non_special_indices = _encode_non_special(split);
                    indices.insert(indices.end(), non_special_indices.begin(), non_special_indices.end());
                }
            }
        } else {
            indices = _encode_non_special(input);
        }

        return indices;
    }

    std::string decode(const std::vector<int>& indices) {
        std::string decoded;

        for (int id : indices) {
            if (id_to_special.find(id) != id_to_special.end()) {
                decoded += id_to_special[id];
            } else {
                decoded += id_to_token[id];
            }
        }

        return decoded;
    }

    int vocab_size() const {
        return next_id;
    }

private:
    std::vector<int> _encode_non_special(const std::string& input) {
        std::vector<int> indices = string_to_byte(input, "utf-8");
        
        bool changes_made;
        do {
            changes_made = false;
            for (size_t i = 0; i < indices.size() - 1; ++i) {
                std::pair<int, int> pair = {indices[i], indices[i + 1]};
                if (pairs.find(pair) != pairs.end()) {
                    indices[i] = pairs[pair];
                    indices.erase(indices.begin() + i + 1);
                    changes_made = true;
                }
            }
        } while (changes_made);

        return indices;
    }

    int max_vocab_size;
    std::unordered_map<std::pair<int, int>, int, pair_hash> pairs;
    std::unordered_map<int, std::string> id_to_token;
    int next_id;
    std::unordered_map<std::string, int> special_to_id;
    std::unordered_map<int, std::string> id_to_special;
};


int main() {
    try {
        std::cout << "Opening file...\n" << std::endl;
        std::ifstream file("data.txt");
        if (!file.is_open()) {
            std::cerr << "Error opening data.txt\n" << std::endl;
            return 1;
        }
        std::cout << "File opened successfully\n" << std::endl;

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string corpus = buffer.str();
        file.close();

        std::cout << "Corpus size: " << corpus.size() << " characters\n" << std::endl;

        if (corpus.empty()) {
            std::cerr << "Error: data.txt is empty\n" << std::endl;
            return 1;
        }

        BPETokenizer tokenizer(MAX_VOCAB_SIZE);

        std::string verbose_input;
        bool verbose = false;
        
        while (true) {
            std::cout << "Print merge information? (y/n): ";
            
            std::getline(std::cin, verbose_input);

            if (verbose_input == "y" || verbose_input == "Y") {
                verbose = true;
                break;

            } else if (verbose_input == "n" || verbose_input == "N") {
                verbose = false;
                break;

            } else {
                std::cout << "\nInvalid input. Please enter 'y' or 'n'\n" << std::endl;
            }
        }

        tokenizer.train(corpus, false, verbose);

        tokenizer.register_special_token("<|endoftext|>");

        std::string input;

        while (true) {
            std::cout << "\nEnter text to encode (or 'q' to quit): ";
            std::getline(std::cin, input);

            if (input == "q") {
                break;
            }

            std::vector<int> encoded = tokenizer.encode(input);

            std::cout << "Encoded: ";
            for (int id : encoded) {
                std::cout << id << " ";
            }
            std::cout << std::endl;

            std::string decoded = tokenizer.decode(encoded);
            std::cout << "Decoded: " << decoded << std::endl;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}