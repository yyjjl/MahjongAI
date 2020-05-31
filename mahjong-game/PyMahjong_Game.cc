#include <cstring>
#include <sstream>

#include "PyMahjong_Game.h"

bool MahjongGame::InitTiles(PlayerID player_id, pybind11::list &tiles, int flower_count) {
    rest_wall_count -= flower_count;

    auto player = &all_players[player_id];

    player->flower_count = flower_count;
    for (auto &tile : tiles)
        player->Add(IndexToTile(tile.cast<int>()));

    return tiles.size() == 13;
}

void MahjongGame::InitSeat(Wind prevalent_wind, PlayerID master_id) {
    for (int i = 0; i < 4; ++i) { // 圈风都一样
        all_players[i].prevalent_wind = prevalent_wind;
        // 庄家门风是东风
        all_players[i].seat_wind = static_cast<Wind>((i - master_id + 4) % 4);
    }
}

void MahjongGame::Draw(PlayerID player_id, int tile_index) {
    Tile tile = IndexToTile(tile_index);

    auto &message = history[turn_ID++];
    message.type = Message::kDraw;
    message.player_id = player_id;
    message.info_tile = tile;

    if (tile != kUnknownTile) {
        if (all_players[player_id].win_tile != kEmptyTile)
            throw std::runtime_error("STRANGE DRAW ACTION");

        all_players[player_id].Add(tile, true /* is_temporary */);
    }

    --rest_wall_count;
}

void MahjongGame::AddFlower(PlayerID player_id) {
    auto &message = history[turn_ID++];
    message.type = Message::kExtraFlower;
    message.player_id = player_id;
    all_players[player_id].flower_count++;

    --rest_wall_count;
}

bool MahjongGame::Play(PlayerID player_id, int output_tile_index) {
    Tile output_tile = IndexToTile(output_tile_index);

    auto &message = history[turn_ID++];
    message.type = Message::kPlay;
    message.player_id = player_id;
    message.output_tile = output_tile;

    PoolPush(output_tile, player_id); // 更新牌池

    return all_players[player_id].Remove(output_tile);
}

bool MahjongGame::Pung(PlayerID player_id, int output_tile_index) {
    Tile output_tile = IndexToTile(output_tile_index);

    // 上一次必须打出了牌
    if (turn_ID == 0 || history[turn_ID - 1].output_tile == kEmptyTile)
        throw std::runtime_error("FAILED TO PUNG");

    auto &message = history[turn_ID++];
    message.type = Message::kPung;
    message.player_id = player_id;
    message.output_tile = output_tile;

    auto player = &all_players[player_id];
    auto [tile, last_player_id] = PoolPop();

    PoolPush(output_tile, player_id); // 更新牌池

    return player->AddPungPack(tile, last_player_id) && player->Remove(output_tile);
}

bool MahjongGame::Chow(PlayerID player_id, int info_tile_index, int output_tile_index) {
    Tile output_tile = IndexToTile(output_tile_index);
    Tile info_tile = IndexToTile(info_tile_index);

    // 上一次必须打出了牌
    if (turn_ID == 0 || history[turn_ID - 1].output_tile == kEmptyTile)
        throw std::runtime_error("FAILED TO CHOW");

    auto &message = history[turn_ID++];
    message.type = Message::kChow;
    message.player_id = player_id;
    message.info_tile = info_tile;
    message.output_tile = output_tile;

    auto player = &all_players[player_id];
    auto [input_tile, last_player_id] = PoolPop();

    if (last_player_id != (static_cast<int>(player_id) + 3) % 4) // 只能上家
        throw std::runtime_error("FAILED TO CHOW (ONLY UPSTREAM)");

    PoolPush(output_tile, player_id); // 更新牌池

    return player->AddChowPack(info_tile, input_tile) && player->Remove(output_tile);
}

bool MahjongGame::Kong(PlayerID player_id, int tile_index, bool is_hidden) {
    Tile tile = IndexToTile(tile_index);

    auto &last_message = history[turn_ID - 1];
    auto &message = history[turn_ID++];
    message.type = Message::kKong;
    message.player_id = player_id;
    message.info_tile = tile;

    auto player = &all_players[player_id];
    bool success = false;
    if (is_hidden) { // 暗杠, 麻将对局数据中, 任何时候都可以
        if (tile != kUnknownTile) {
            player->Remove(tile);
            player->Add(tile, true /* is_temporary */); // 把牌放到最后
        }
        success = player->AddKongPack(kUnknownTile, player_id, false);
    } else { // 直杠
        if (last_message.output_tile == kEmptyTile)
            throw std::runtime_error("FAILED TO KONG, LAST OUTPUT IS EMPTY");
        auto [tile, last_player_id] = PoolPop();
        success = player->AddKongPack(tile, last_player_id, false);
    }

    return success;
}

bool MahjongGame::AddKong(PlayerID player_id, int info_tile_index) {
    Tile info_tile = IndexToTile(info_tile_index);

    auto &message = history[turn_ID++];
    message.type = Message::kExtraKong;
    message.player_id = player_id;
    message.info_tile = info_tile;

    // 补杠, 麻将对局数据中, 任何时候都可以
    if (turn_ID == 0)
        throw std::runtime_error("FAILED TO ADD KONG");

    return all_players[player_id].AddKongPack(info_tile, player_id, true /* from_pung_pack */);
}

FanResult MahjongGame::Win(PlayerID player_id, int tile_index) {
    Tile tile = IndexToTile(tile_index);

    ComputeTilesCount(player_id);

    auto player = &all_players[player_id];

    bool is_selfdrawn = player->win_tile != kEmptyTile;
    if (is_selfdrawn && tile != player->win_tile)
        throw std::runtime_error("TILES MISMATCH");

    if (!is_selfdrawn) {
        if (tile == kEmptyTile || player->win_tile != kEmptyTile)
            throw std::runtime_error("STRANGE CONDITION");
        player->Add(tile, true /* is_temporary */);
    }

    bool is_kong = false;
    if (turn_ID > 1) {
        auto type = history[turn_ID - 1].type;
        auto last_type = history[turn_ID - 2].type;
        is_kong = (type == Message::kKong || type == Message::kExtraKong) ||
                  (type == Message::kDraw &&
                   (last_type == Message::kKong || last_type == Message::kExtraKong));
    }
    bool is_last_tile = rest_wall_count == 0;
    bool is_4th_tile = rest_tiles_count[tile] == 0 && player->hand_tiles_count[tile] == 1;
    auto result = Player_ComputeFan(*player, is_selfdrawn, is_kong, is_last_tile, is_4th_tile);

    if (!is_selfdrawn)
        player->Remove(tile);

    return result;
}

int MahjongGame::TryWin(PlayerID player_id, bool apply_change) {
    ComputeTilesCount(player_id);

    auto player = &all_players[player_id];
    auto win_tile = player->win_tile;

    bool is_selfdrawn = win_tile != kEmptyTile;
    if (!is_selfdrawn) {
        auto &last_message = history[turn_ID - 1];
        auto last_player_id = last_message.player_id;

        if (last_player_id == player_id)
            return ERROR_NOT_WIN;

        if (last_message.type == Message::kExtraKong) { // 抢杠和
            win_tile = last_message.info_tile;
            if (win_tile == kEmptyTile)
                return ERROR_NOT_WIN;

            if (apply_change) { // 拆掉
                auto &last_player = all_players[last_player_id];
                auto &hand_tiles = last_player.hand_tiles;
                int index = 0;
                while (index < hand_tiles.pack_count) {
                    auto pack = hand_tiles.fixed_packs[index];
                    if (mahjong::pack_get_type(pack) == PACK_TYPE_KONG &&
                        mahjong::pack_get_tile(pack) == win_tile)
                        break;
                    index += 1;
                }
                assert(index != hand_tiles.pack_count);

                last_player.all_tiles_count[win_tile]--; // 牌被拿走了

                auto &pack = hand_tiles.fixed_packs[index]; // 修改杠牌为碰
                pack = mahjong::make_pack(mahjong::pack_get_offer(pack), PACK_TYPE_PUNG, win_tile);
            }
        } else {
            win_tile = last_message.output_tile;
            if (win_tile == kEmptyTile)
                return ERROR_NOT_WIN;
        }

        //  把牌拿到手里
        player->Add(win_tile, true /* is_temporary */);
        if (apply_change)
            PoolPop();
    }

    bool is_kong = false;
    if (turn_ID > 1) {
        auto type = history[turn_ID - 1].type;
        auto last_type = history[turn_ID - 2].type;
        is_kong = (type == Message::kKong || type == Message::kExtraKong) ||
                  (type == Message::kDraw &&
                   (last_type == Message::kKong || last_type == Message::kExtraKong));
    }
    bool is_last_tile = rest_wall_count == 0;
    bool is_4th_tile = rest_tiles_count[win_tile] == 0 && player->hand_tiles_count[win_tile] == 1;
    auto ret = player->ComputeFan(is_selfdrawn, is_kong, is_last_tile, is_4th_tile);

    if (!is_selfdrawn && !apply_change)
        player->Remove(win_tile);

    return ret;
}

void MahjongGame::Clear() {
    turn_ID = 0;
    rest_wall_count = 92;
    tiles_pool_size = 0;

    std::memset(tiles_count_in_pool.data(), 0, sizeof(tiles_count_in_pool));
    std::memset(all_players, 0, sizeof(all_players));
    for (int i = 0; i < 4; ++i) {
        all_players[i].index = i;
        all_players[i].win_tile = mahjong::TILE_TABLE_SIZE;
    }
}

bool MahjongGame::CanAddKong(PlayerID player_id, int tile_index) const {
    Tile tile = IndexToTile(tile_index);

    auto &hand_tiles = all_players[player_id].hand_tiles;
    int index = 0;
    while (index < hand_tiles.pack_count) {
        auto pack = hand_tiles.fixed_packs[index];
        if (mahjong::pack_get_type(pack) == PACK_TYPE_PUNG && mahjong::pack_get_tile(pack) == tile)
            break;
        index += 1;
    }

    return index < hand_tiles.pack_count;
}

void MahjongGame::PoolPush(Tile tile, PlayerID play_id) {
    tiles_pool_stack[tiles_pool_size++] = {tile, play_id};
    ++tiles_count_in_pool[tile];
}

MahjongGame::Item MahjongGame::PoolPop() {
    assert(tiles_pool_size > 0);
    --tiles_pool_size;
    auto item = tiles_pool_stack[tiles_pool_size];
    --tiles_count_in_pool[std::get<0>(item)];
    return item;
}

void MahjongGame::ComputeTilesCount(PlayerID player_id) {
    std::fill(rest_tiles_count.begin(), rest_tiles_count.end(), 0);

    for (int i = 0; i < 34; ++i) {
        Tile tile = mahjong::all_tiles[i];
        auto &count = rest_tiles_count[tile];
        count = 4 - tiles_count_in_pool[tile];

        for (int j = 0; j < 4; ++j) {
            count -= all_players[j].all_tiles_count[tile];
            if (j != player_id) // 除了自己别人手牌看不到
                count += all_players[j].hand_tiles_count[tile];
        }
        assert(count >= 0 && count <= 4);
    }
}

std::string MahjongGame::Print() {
    static const char *seat_strings[]{"东", "南", "西", "北"};

    std::ostringstream os;

    os << "========================================\n"
       << "轮数: " << turn_ID + 1 << " 剩余: " << rest_wall_count << '\n';
    for (int i = 0; i < 4; ++i) {
        auto player = &all_players[i];
        player->Print(os << seat_strings[i] << '\n');
    }

    os << "\n牌池:";
    for (int i = 0; i < tiles_pool_size; ++i) {
        auto [tile, player_id] = tiles_pool_stack[i];
        WriteTile(os << " [", tile);
        os << "@" << seat_strings[static_cast<int>(player_id)] << "]";
    }
    os << '\n';

    return os.str();
}
