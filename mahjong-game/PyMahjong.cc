#include <array>
#include <iterator>
#include <sstream>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "PyMahjong_Game.h"
#include "shanten.h"

using LongArray = pybind11::array_t<long>;
using BoolArray = pybind11::array_t<bool>;
using FloatArray = pybind11::array_t<float>;

const int g_tile_to_index[] = {0,  35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
                               35, 35, 1,  2,  3,  4,  5,  6,  7,  8,  9,  35, 35, 35, 35,
                               35, 35, 35, 10, 11, 12, 13, 14, 15, 16, 17, 18, 35, 35, 35,
                               35, 35, 35, 35, 19, 20, 21, 22, 23, 24, 25, 26, 27, 35, 35,
                               35, 35, 35, 35, 35, 28, 29, 30, 31, 32, 33, 34, 35};

int TileToIndex(Tile tile) { // 1 - 34, 0 是 padding
    if (tile > kEmptyTile)
        throw std::runtime_error("INVALID TILE " + std::to_string(static_cast<int>(tile)));
    return g_tile_to_index[tile];
}

pybind11::tuple Player_NearbyCounts(const Player &self, int tile_index) {
    if (tile_index < 1 || tile_index > 27)
        pybind11::make_tuple(0, 0, 0, 0);

    auto tile = IndexToTile(tile_index);
    return pybind11::make_tuple(self.hand_tiles_count[tile - 2], self.hand_tiles_count[tile - 1],
                                self.hand_tiles_count[tile + 1], self.hand_tiles_count[tile + 2]);
}

FloatArray Player_TilesMask(const Player &self) {
    FloatArray mask(35);
    auto ptr = static_cast<float *>(mask.request().ptr);
    std::fill_n(ptr, mask.size(), 0);

    for (int i = 0; i < 34; ++i) {
        Tile tile = mahjong::all_tiles[i];
        ptr[i + 1] = self.hand_tiles_count[tile] > 0;
    }
    return mask;
}

LongArray TilesCountToNumpy(const TilesCount &tiles_count) {
    LongArray counts(36);

    auto ptr = static_cast<long *>(counts.request().ptr);
    for (int i = 0; i < 34; ++i)
        ptr[i] = tiles_count[mahjong::all_tiles[i]];
    ptr[34] = ptr[35] = 0;

    counts.resize({4, 9});
    return counts;
}

LongArray MahjongGame_AllPacks(const MahjongGame &self, PlayerID player_id) {
    LongArray packs(4 * 4 * 3);

    auto ptr = static_cast<long *>(packs.request().ptr);
    std::fill_n(ptr, packs.size(), 0);

    // 依次是自己, 下家, 对家, 上家
    for (int i = 0; i < 4; ++i) {
        auto player = &self.all_players[(player_id + i) % 4];
        auto pack_count = player->hand_tiles.pack_count;
        auto &fix_packs = player->hand_tiles.fixed_packs;
        assert(pack_count <= 4);
        for (int j = 0; j < pack_count; ++j) {
            auto pack = fix_packs[j];
            auto index = (i * 4 + j) * 3;
            ptr[index] = mahjong::pack_get_offer(pack);
            ptr[index + 1] = mahjong::pack_get_type(pack);
            // 自己的牌或者别人的明牌才看得到
            ptr[index + 2] = (mahjong::is_pack_melded(pack) || player->index == player_id)
                                 ? TileToIndex(mahjong::pack_get_tile(pack))
                                 : 0;
        }
    }

    packs.resize({4, 4, 3});
    return packs;
}

LongArray MahjongGame_TilesPoolCounts(const MahjongGame &self, PlayerID player_id) {
    LongArray tiles_pool(4 * 36);
    auto ptr = static_cast<long *>(tiles_pool.request().ptr);
    std::fill_n(ptr, tiles_pool.size(), 0);

    PlayerID tile_player_id;
    Tile tile;
    for (int i = self.tiles_pool_size - 1; i >= 0; --i) {
        std::tie(tile, tile_player_id) = self.tiles_pool_stack[i];
        int position = (tile_player_id - player_id + 4) % 4;
        ptr[position * 36 + TileToIndex(tile) - 1]++;
    }

    tiles_pool.resize({4, 4, 9});
    return tiles_pool;
}

LongArray MahjongGame_TilesPool(const MahjongGame &self, PlayerID player_id, int size) {
    LongArray tiles_pool(4 * size);
    auto ptr = static_cast<long *>(tiles_pool.request().ptr);
    std::fill_n(ptr, tiles_pool.size(), 0);

    // 依次是自己, 下家, 对家, 上家
    int counts[4]{0, 0, 0, 0};
    PlayerID tile_player_id;
    Tile tile;
    for (int i = self.tiles_pool_size - 1; i >= 0; --i) {
        std::tie(tile, tile_player_id) = self.tiles_pool_stack[i];
        int position = (tile_player_id - player_id + 4) % 4;
        int &count = counts[position];
        if (count >= size)
            continue;
        ptr[position * size + count] = TileToIndex(tile);
        ++count;
    }

    tiles_pool.resize({4, size});
    return tiles_pool;
}

pybind11::tuple MahjongGame_ViewOfPlayer(MahjongGame &self, PlayerID player_id) {
    self.ComputeTilesCount(player_id);
    return pybind11::make_tuple(TilesCountToNumpy(self.rest_tiles_count),
                                TilesCountToNumpy(self.all_players[player_id].hand_tiles_count),
                                TilesCountToNumpy(self.all_players[player_id].all_tiles_count));
}

pybind11::tuple Player_ShanTen(const Player &player) {
    BoolArray useful_tiles(34);
    auto ptr = static_cast<bool *>(useful_tiles.request().ptr);
    std::fill_n(ptr, useful_tiles.size(), false);

    mahjong::useful_table_t useful_table;

    int shanten1 = mahjong::basic_form_shanten(player.hand_tiles.standing_tiles,
                                               player.hand_tiles.tile_count, &useful_table);
    for (int i = 0; i < 34; ++i)
        ptr[i] |= useful_table[mahjong::all_tiles[i]];

    int shanten2 = mahjong::seven_pairs_shanten(player.hand_tiles.standing_tiles,
                                                player.hand_tiles.tile_count, &useful_table);
    for (int i = 0; i < 34; ++i)
        ptr[i] |= useful_table[mahjong::all_tiles[i]];

    int shanten3 = mahjong::thirteen_orphans_shanten(player.hand_tiles.standing_tiles,
                                                     player.hand_tiles.tile_count, &useful_table);
    for (int i = 0; i < 34; ++i)
        ptr[i] |= useful_table[mahjong::all_tiles[i]];

    return pybind11::make_tuple(shanten1, shanten2, shanten3, useful_tiles);
}

PYBIND11_MODULE(PyMahjong, m) {
    using namespace pybind11;
    using namespace pybind11::literals;

    m.doc() = "MahJong"; // optional module docstring
    m.attr("TILE_SUIT_BAMBOO") = TILE_SUIT_BAMBOO;
    m.attr("TILE_SUIT_CHARACTERS") = TILE_SUIT_CHARACTERS;
    m.attr("TILE_SUIT_DOTS") = TILE_SUIT_DOTS;
    m.attr("TILE_SUIT_HONORS") = TILE_SUIT_HONORS;
    m.attr("kUnknownTile") = kUnknownTile;

    m.def("tiles_count_to_numpy", TilesCountToNumpy);

    enum_<Message::Type>(m, "MessageType") //
        .value("kInitSeat", Message::kInitSeat)
        .value("kInitTiles", Message::kInitTiles)
        .value("kSelfDraw,  ", Message::kSelfDraw)
        .value("kExtraFlower", Message::kExtraFlower)
        .value("kDraw", Message::kDraw)
        .value("kPlay", Message::kPlay)
        .value("kPung", Message::kPung)
        .value("kChow", Message::kChow)
        .value("kKong", Message::kKong)
        .value("kExtraKong", Message::kExtraKong)
        .value("kWin", Message::kWin)
        .value("kPass", Message::kPass);

    class_<Message>(m, "Message") //
        .def_readonly("type", &Message::type)
        .def_readonly("player_id", &Message::player_id)
        .def_property_readonly("info_tile_index",
                               [](const Message &self) { return TileToIndex(self.info_tile); })
        .def_property_readonly("output_tile_index",
                               [](const Message &self) { return TileToIndex(self.output_tile); })
        .def_readonly("men_feng", &Message::men_feng)
        .def_readonly("quan_feng", &Message::quan_feng);

    enum_<Wind>(m, "Wind") //
        .value("kEast", Wind::EAST)
        .value("kSouth", Wind::SOUTH)
        .value("kWest", Wind::WEST)
        .value("kNorth", Wind::NORTH);

    class_<Player>(m, "Player") //
        .def(init([]() {
            Player self;
            std::memset(&self, 0, sizeof(Player));
            self.win_tile = mahjong::TILE_TABLE_SIZE;
            return self;
        }))
        .def_readwrite("flower_count", &Player::flower_count)
        .def_property_readonly("win_tile_index",
                               [](const Player &self) { return TileToIndex(self.win_tile); })
        .def_readwrite("seat_wind", &Player::seat_wind)
        .def_readwrite("prevalent_wind", &Player::prevalent_wind)
        .def("add",
             [](Player &self, int tile_index, bool is_temporary) {
                 self.Add(IndexToTile(tile_index), is_temporary);
             },
             "tile"_a, "is_temporary"_a = false)
        // .def("add_pung_pack", &Player::AddPungPack)
        // .def("add_kong_pack", &Player::AddKongPack)
        // .def("add_chow_pack", &Player::AddChowPack)
        .def("remove",
             [](Player &self, int tile_index) { return self.Remove(IndexToTile(tile_index)); })
        .def("compute_fan", &Player::ComputeFan, //
             "is_selfdrawn"_a, "is_kong"_a, "is_last_tile"_a, "is_4th_tile"_a)
        .def("compute_fan_slow", &Player_ComputeFan, //
             "is_selfdrawn"_a, "is_kong"_a, "is_last_tile"_a, "is_4th_tile"_a)
        .def("is_upstream", &Player::IsUpstream)
        .def("tiles_mask", &Player_TilesMask)
        .def("count",
             [](const Player &self, int tile_index, bool ignore_packs) {
                 Tile tile = IndexToTile(tile_index);
                 if (ignore_packs)
                     return self.hand_tiles_count[tile];
                 return self.all_tiles_count[tile];
             },
             "tile_index"_a, "ignore_packs"_a)
        .def("nearby_counts", &Player_NearbyCounts)
        .def("shanten", &Player_ShanTen)
        .def("__str__", [](const Player &self) {
            std::ostringstream os;
            self.Print(os);
            return os.str();
        });

    class_<MahjongGame>(m, "Game") //
        .def(init())
        .def_readonly("turn_ID", &MahjongGame::turn_ID)
        .def_readonly("rest_wall_count", &MahjongGame::rest_wall_count)
        .def("copy", [](const MahjongGame &self) { return MahjongGame(self); })
        .def("history",
             [](const MahjongGame &self, int index) {
                 if (index < 0 || index >= self.turn_ID)
                     throw std::out_of_range( //
                         "out of range [0, " + std::to_string(self.turn_ID) + ")");
                 return self.history[index];
             })
        .def("player",
             [](const MahjongGame &self, int index) {
                 if (index < 0 || index >= 4)
                     throw std::out_of_range("out of range [0, 4)");
                 return self.all_players[index];
             })
        .def("compute_tiles_count", &MahjongGame::ComputeTilesCount)

        .def("init_tiles", &MahjongGame::InitTiles)
        .def("init_seat", &MahjongGame::InitSeat)
        .def("play", &MahjongGame::Play)
        .def("draw", &MahjongGame::Draw)
        .def("chow", &MahjongGame::Chow)
        .def("pung", &MahjongGame::Pung)
        .def("kong", &MahjongGame::Kong, "player_id"_a, "tile_index"_a = kEmptyTileIndex,
             "is_hidden"_a = false)
        .def("win", &MahjongGame::Win)
        .def("add_flower", &MahjongGame::AddFlower)
        .def("add_kong", &MahjongGame::AddKong)

        .def("try_win", &MahjongGame::TryWin, "player_id"_a, "apply_change"_a = false)
        .def("can_add_kong", &MahjongGame::CanAddKong)
        .def("clear", &MahjongGame::Clear)
        .def("all_packs", &MahjongGame_AllPacks)
        .def("tiles_pool", &MahjongGame_TilesPool)
        .def("tiles_pool_counts", &MahjongGame_TilesPoolCounts)
        .def("view_of_player", &MahjongGame_ViewOfPlayer)
        .def("__str__", &MahjongGame::Print);
}
