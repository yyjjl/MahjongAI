#ifndef MAHJONG_PLAYER_H
#define MAHJONG_PLAYER_H

#include <array>
#include <cassert>

#include "fan_calculator.h"

#include "message.h"

using Wind = mahjong::wind_t;
using TilesCount = std::array<uint8_t, mahjong::TILE_TABLE_SIZE>;

struct Player : public mahjong::calculate_param_t {
    PlayerID index;
    TilesCount all_tiles_count;
    TilesCount hand_tiles_count;

    void Add(Tile tile, bool is_temporary = false);

    bool IsUpstream(PlayerID id) const { return ((index - id + 4) % 4) == 1; }

    bool AddPungPack(Tile tile, PlayerID offer_id);
    bool AddChowPack(Tile center_tile, Tile input_tile);
    bool AddKongPack(Tile tile, PlayerID offer_id, bool from_pung_pack);

    bool Remove(Tile tile);

    int ComputeFan(bool is_selfdrawn, bool is_kong, bool is_last_tile, bool is_4th_tile);

    void Print(std::ostream &os) const;
};

#endif /* MAHJONG_PLAYER_H */

// Local Variables:
// mode: c++
// End:
