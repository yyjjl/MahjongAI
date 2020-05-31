#ifndef PYMAHJONG_H
#define PYMAHJONG_H

#include "message.h"
#include "player.h"
#include "pybind11/pybind11.h"

using FanResult = std::vector<std::tuple<const char *, int>>;

const int kEmptyTileIndex = 35;

inline Tile IndexToTile(int index) {
    if (index < 0 || index > 35)
        throw std::runtime_error("INVALID TILE INDEX " + std::to_string(index));
    if (index == 0)
        return kUnknownTile;
    if (index == kEmptyTileIndex)
        return kEmptyTile;
    return mahjong::all_tiles[index - 1];
}

struct MahjongGame {
    using Item = std::tuple<Tile, PlayerID>;

    TilesCount rest_tiles_count; // 剩余牌计数
    int rest_wall_count;         // 剩余牌数

    TilesCount tiles_count_in_pool; // 牌池
    Item tiles_pool_stack[150];
    int tiles_pool_size;

    Player all_players[4];

    int turn_ID;
    Message history[1000];

    void InitSeat(Wind prevalent_wind, PlayerID master_id);
    bool InitTiles(PlayerID player_id, pybind11::list &tiles, int flower_count);

    void Draw(PlayerID player_id, int tile_index);
    void AddFlower(PlayerID player_id);
    bool Play(PlayerID player_id, int output_tile_index);
    bool Chow(PlayerID player_id, int info_tile_index, int output_tile_index);
    bool Pung(PlayerID player_id, int output_tile_index);
    bool Kong(PlayerID player_id, int tile_index = kEmptyTileIndex, bool is_hidden = false);
    bool AddKong(PlayerID player_id, int info_tile_index);

    bool CanAddKong(PlayerID player_id, int tile_index) const;

    FanResult Win(PlayerID player_id, int tile_index = kEmptyTileIndex);

    int TryWin(PlayerID player_id, bool apply_change = false);

    MahjongGame() { Clear(); }
    void Clear();
    std::string Print();

    void ComputeTilesCount(PlayerID player_id);

  private:
    void PoolPush(Tile tile, PlayerID play_id);
    Item PoolPop();
};

FanResult Player_ComputeFan(Player &self,                    //
                            bool is_selfdrawn, bool is_kong, //
                            bool is_last_tile, bool is_4th_tile);

#endif /* PYMAHJONG_H */

// Local Variables:
// mode: c++
// End:
