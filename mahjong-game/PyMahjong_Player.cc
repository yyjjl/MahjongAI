#include <cstring>
#include <vector>

#include "PyMahjong_Game.h"

bool CheckWin(const TilesCount &tiles_count);

FanResult Player_ComputeFan(Player &self,                    //
                            bool is_selfdrawn, bool is_kong, //
                            bool is_last_tile, bool is_4th_tile) {
    // if (!CheckWin(self.hand_tiles_count))
    //     throw std::runtime_error("ERROR_NOT_WIN_CHECK");

    self.win_flag = 0;
    if (is_selfdrawn)
        self.win_flag |= WIN_FLAG_SELF_DRAWN;
    if (is_last_tile)
        self.win_flag |= WIN_FLAG_WALL_LAST;
    if (is_4th_tile)
        self.win_flag |= WIN_FLAG_4TH_TILE;
    if (is_kong)
        self.win_flag |= WIN_FLAG_ABOUT_KONG;

    mahjong::fan_table_t fan_table;
    std::memset(&fan_table, 0, sizeof(mahjong::fan_table_t));
    int max_fan = mahjong::calculate_fan(&self, &fan_table);

    if (max_fan > 0 && max_fan - self.flower_count < 8) // 为达到 8 番
        throw std::runtime_error("ERROR_NOT_WIN: less than 8");
    if (max_fan == -1)
        throw std::runtime_error("ERROR_WRONG_TILES_COUNT");
    else if (max_fan == -2)
        throw std::runtime_error("ERROR_TILE_COUNT_GREATER_THAN_4");
    else if (max_fan == -3)
        throw std::runtime_error("ERROR_NOT_WIN");

    std::vector<std::tuple<const char *, int>> result;
    for (int i = 1; i < mahjong::FAN_TABLE_SIZE; ++i) {
        if (fan_table[i] == 0)
            continue;
        result.push_back({mahjong::fan_name[i], //
                          mahjong::fan_value_table[i] * fan_table[i]});
    }

    return result;
}
