#ifndef MESSAGE_H
#define MESSAGE_H

#include <iosfwd>

#include "tile.h"

using PlayerID = int8_t;
using Tile = mahjong::tile_t;
using TilePack = mahjong::pack_t;

const Tile kEmptyTile = mahjong::TILE_TABLE_SIZE; // 占位牌
const Tile kUnknownTile = 0;                       // 未知的牌, 比如对手的手牌
constexpr inline bool IsEmptyTile(Tile tile) { return tile == kEmptyTile; }

struct Message {
    enum Type : uint8_t {
        kInitSeat = 0,
        kInitTiles,
        kSelfDraw,    // 2 摸到牌
        kExtraFlower, // 3 补花
        kDraw,        // 4 其他玩家摸牌
        kPlay,        // 5 打出牌
        kPung,        // 6 碰
        kChow,        // 7 吃
        kKong,        // 8 杠
        kExtraKong,   // 9 补杠
        kWin,         // 10 和
        kPass,        // 11
    };

    Type type;
    PlayerID player_id = -1;
    union {
        Tile info_tile = kEmptyTile;
        Tile men_feng;
    };
    union {
        Tile output_tile = kEmptyTile;
        Tile quan_feng;
    };
};

struct Request : public Message {
    void Read(std::istream &in);
};

struct Response : public Message {
    void Read(std::istream &in, Type request_type);
    void Write(std::ostream &os) const;
};

Tile ReadTile(std::istream &in);
void WriteTile(std::ostream &os, Tile tile);

#endif /* MESSAGE_H */

// Local Variables:
// mode: c++
// End:
