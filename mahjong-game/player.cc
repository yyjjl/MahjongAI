#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

#include "player.h"

void Player::Add(Tile tile, bool is_temporary) {
    assert(tile != kEmptyTile);
    all_tiles_count[tile]++;
    hand_tiles_count[tile]++;
    if (is_temporary) {
        assert(win_tile == kEmptyTile);
        win_tile = tile;
    } else
        hand_tiles.standing_tiles[hand_tiles.tile_count++] = tile;
}

bool Player::AddPungPack(Tile tile, PlayerID offer_id) { // 碰牌
    bool success = Remove(tile) && Remove(tile);         // 首先移除手中的两张牌
    all_tiles_count[tile] += 3;                          // 当前牌数量 + 3

    // 123 分别来表示是 上家 / 对家 / 下家 供的
    auto offer = (index - offer_id + 4) % 4;
    assert(offer != 0);

    auto &pack_count = hand_tiles.pack_count;
    hand_tiles.fixed_packs[pack_count++] = mahjong::make_pack(offer, PACK_TYPE_PUNG, tile);
    return success;
}

bool Player::AddChowPack(Tile center_tile, Tile input_tile) { // 吃牌
    bool success = true;
    for (int i = -1; i <= 1; ++i) {
        Tile tile = center_tile + i;
        if (tile != input_tile)
            success = Remove(tile) && success;
        all_tiles_count[tile]++;
    }

    // 这里用 123 分别来表示第几张是上家供的
    auto offer = input_tile - center_tile + 2;
    assert(offer >= 1 && offer <= 3);

    auto &pack_count = hand_tiles.pack_count;
    hand_tiles.fixed_packs[pack_count++] = mahjong::make_pack(offer, PACK_TYPE_CHOW, center_tile);
    return success;
}

bool Player::AddKongPack(Tile tile, PlayerID offer_id, bool from_pung_pack) { // 杠
    auto &pack_count = hand_tiles.pack_count;

    if (from_pung_pack) { // 补杠
        uint index = 0;
        for (; index < pack_count; ++index) {
            auto &pack = hand_tiles.fixed_packs[index];
            if (mahjong::pack_get_type(pack) == PACK_TYPE_PUNG &&
                mahjong::pack_get_tile(pack) == tile)
                break;
        }
        // 一定要找到, 补杠来自碰, 碰是明牌
        assert(index != pack_count);

        all_tiles_count[tile]++;
        auto &pack = hand_tiles.fixed_packs[index]; // 修改碰牌为杠
        pack = mahjong::make_pack(mahjong::pack_get_offer(pack), PACK_TYPE_KONG, tile);
        // 补杠是自摸的牌, 所以要删去
        return Remove(tile);
    }

    bool success = true;
    auto offer = (index - offer_id + 4) % 4;
    if (offer == 0 && win_tile != kEmptyTile) {
        tile = win_tile;        // 暗杠, 自己摸到牌
        success = Remove(tile); // 第 4 张牌是摸到的, 也要 Remove
    }
    success = Remove(tile) && Remove(tile) && Remove(tile);
    all_tiles_count[tile] += 4;

    hand_tiles.fixed_packs[pack_count++] = mahjong::make_pack(offer, PACK_TYPE_KONG, tile);
    return success;
}

bool Player::Remove(Tile tile) {
    if (win_tile == tile) {
        win_tile = kEmptyTile; // 打出摸到的牌
        all_tiles_count[tile]--;
        hand_tiles_count[tile]--;
        return true;
    }

    uint index = 0;
    auto &tile_count = hand_tiles.tile_count;
    auto &standing_tiles = hand_tiles.standing_tiles;

    for (; index < tile_count; ++index)
        if (standing_tiles[index] == tile)
            break;
    if (index == tile_count) // 没找到要打出的牌
        return false;

    all_tiles_count[tile]--;
    hand_tiles_count[tile]--;
    if (IsEmptyTile(win_tile)) { // 没有摸到的牌
        tile_count--;            // 和最后一张牌交换
        std::swap(standing_tiles[index], standing_tiles[tile_count]);
    } else { // 把摸到的牌放入打出的牌的位置
        standing_tiles[index] = win_tile;
        win_tile = kEmptyTile;
    }
    return true;
}

bool CheckWinInternal(uint8_t counts[12], int total_count) {
    int first_index = 0;
    while (first_index < 9 && counts[first_index] == 0)
        ++first_index;

    if (first_index == 9)
        return true;

    auto &first_count = counts[first_index];
    auto &second_count = counts[first_index + 1];
    auto &third_count = counts[first_index + 2];
    if (total_count == 3)
        return first_count == 3 || //
               (first_count == 1 && second_count == 1 && third_count == 1);

    if (total_count == 2)
        return first_count == 2;

    if (total_count == 1)
        return false;

    if (first_count >= 3) {
        first_count -= 3;
        if (CheckWinInternal(counts, total_count - 3))
            return true;
        first_count += 3;
    }

    if (first_count == 2 && total_count % 3 == 2) {
        first_count -= 2;
        if (CheckWinInternal(counts, total_count - 2))
            return true;
        first_count += 2;
    }

    if (first_count >= 1 && second_count >= 1 && third_count >= 1) {
        --first_count, --second_count, --third_count;
        if (CheckWinInternal(counts, total_count - 3))
            return true;
        ++first_count, ++second_count, ++third_count;
    }

    return false;
}

bool CheckWin(const TilesCount &tiles_count) {
    int total_count = 0, pair_count = 0;

    for (int i = 0; i < 34; ++i) {
        int count = tiles_count[mahjong::all_tiles[i]];
        if ((count & 1) == 0)
            pair_count += (count >> 1);
        total_count += count;
    }
    if (total_count == 14 && pair_count == 7) // 七对
        return true;

    if (total_count == 14 && // 十三幺
        tiles_count[mahjong::TILE_1m] >= 1 && tiles_count[mahjong::TILE_9m] >= 1 &&
        tiles_count[mahjong::TILE_1s] >= 1 && tiles_count[mahjong::TILE_9s] >= 1 &&
        tiles_count[mahjong::TILE_1s] >= 1 && tiles_count[mahjong::TILE_9s] >= 1 &&
        tiles_count[mahjong::TILE_E] >= 1 && tiles_count[mahjong::TILE_S] >= 1 &&
        tiles_count[mahjong::TILE_W] >= 1 && tiles_count[mahjong::TILE_N] >= 1 &&
        tiles_count[mahjong::TILE_C] >= 1 && tiles_count[mahjong::TILE_F] >= 1 &&
        tiles_count[mahjong::TILE_P] >= 1)
        return true;

    // 检查字牌
    total_count = pair_count = 0;
    for (Tile tile = mahjong::TILE_E; tile <= mahjong::TILE_P; ++tile) {
        int count = tiles_count[tile];
        pair_count += (count == 2);
        if (count == 1 || pair_count > 1)
            return false;
        total_count += count;
    }

#define CHECK_NUMBER_TILE_STEP_1(suffix)                                                           \
    int total_count_##suffix = 0;                                                                  \
    for (Tile tile = mahjong::TILE_1##suffix; tile <= mahjong::TILE_9##suffix; ++tile)             \
        total_count_##suffix += tiles_count[tile];                                                 \
    {                                                                                              \
        int rest = total_count_##suffix % 3;                                                       \
        pair_count += (rest == 2);                                                                 \
        if (pair_count > 1 || rest == 1)                                                           \
            return false;                                                                          \
    }

#define CHECK_NUMBER_TILE_STEP_2(suffix)                                                           \
    std::copy_n(tiles_count.data() + mahjong::TILE_1##suffix, 12, counts);                         \
    if (!CheckWinInternal(counts, total_count_##suffix))                                           \
        return false;

    // 检查数牌
    uint8_t counts[12];

    CHECK_NUMBER_TILE_STEP_1(m);
    CHECK_NUMBER_TILE_STEP_1(s);
    CHECK_NUMBER_TILE_STEP_1(p);

    CHECK_NUMBER_TILE_STEP_2(m);
    CHECK_NUMBER_TILE_STEP_2(s);
    CHECK_NUMBER_TILE_STEP_2(p);

    return true;
}

int Player::ComputeFan(bool is_selfdrawn, bool is_kong, bool is_last_tile, bool is_4th_tile) {
    // if (!CheckWin(hand_tiles_count))
    //     return -4;

    win_flag = 0;
    if (is_selfdrawn)
        win_flag |= WIN_FLAG_SELF_DRAWN;
    if (is_last_tile)
        win_flag |= WIN_FLAG_WALL_LAST;
    if (is_4th_tile)
        win_flag |= WIN_FLAG_4TH_TILE;
    if (is_kong)
        win_flag |= WIN_FLAG_ABOUT_KONG;

    // mahjong::fan_table_t fan_table;
    // std::memset(&fan_table, 0, sizeof(mahjong::fan_table_t));
    // int max_fan = mahjong::calculate_fan(this, &fan_table);
    int max_fan = mahjong::calculate_fan(this, nullptr);
    if (max_fan > 0 && max_fan - flower_count < 8) // 为达到 8 番
        max_fan = ERROR_NOT_WIN;

    // int fan = 0;
    // for (int i = 1; i < mahjong::FAN_TABLE_SIZE; ++i) {
    //     if (fan_table[i] == 0)
    //         continue;
    //     fan += mahjong::fan_value_table[i] * fan_table[i];
    //     std::cout << mahjong::fan_name[i] << ' ' << mahjong::fan_value_table[i] << ' '
    //               << fan_table[i] << '\n';
    // }

    return max_fan;
}

void Player::Print(std::ostream &os) const {
    static const char *offer_strings[]{"暗", "上", "对", "下"};

    Tile standing_tiles[13];
    auto tile_count = hand_tiles.tile_count;
    std::copy_n(hand_tiles.standing_tiles, tile_count, standing_tiles);
    std::sort(standing_tiles, standing_tiles + tile_count);

    os << "暗牌:";
    for (uint i = 0; i < tile_count; ++i)
        WriteTile(os << " ", standing_tiles[i]);
    os << " |";
    if (win_tile != kEmptyTile)
        WriteTile(os << " ", win_tile);

    os << "\n明牌:";
    for (uint i = 0; i < hand_tiles.pack_count; ++i) {
        auto pack = hand_tiles.fixed_packs[i];
        auto tile = mahjong::pack_get_tile(pack);
        auto offer = mahjong::pack_get_offer(pack);
        auto type = mahjong::pack_get_type(pack);
        os << " (";
        if (type == PACK_TYPE_PUNG) {
            assert(offer >= 1 && offer <= 3);

            os << "刻 " << offer_strings[offer] << " ";
            WriteTile(os, tile);
        } else if (type == PACK_TYPE_CHOW) {
            assert(offer >= 1 && offer <= 3);

            os << "顺";
            for (int i = 1; i <= 3; ++i) {
                os << " ";
                if (i == offer)
                    os << "[";
                WriteTile(os, tile + i - 2);
                if (i == offer)
                    os << "]";
            }
        } else {
            assert(type == PACK_TYPE_KONG && offer >= 0 && offer <= 3);

            os << "杠 " << offer_strings[offer] << " ";
            WriteTile(os, tile);
        }
        os << ")";
    }
    os << " | 花: " << static_cast<int>(flower_count) << "\n计数:";
    for (int tile = 0; tile < kEmptyTile; ++tile) {
        auto count = all_tiles_count[tile];
        if (count == 0)
            continue;
        WriteTile(os << " ", tile);
        os << ":" << static_cast<int>(count);
    }
    os << " | ";
    for (int tile = 0; tile < kEmptyTile; ++tile) {
        auto count = hand_tiles_count[tile];
        if (count == 0)
            continue;
        WriteTile(os << " ", tile);
        os << ":" << static_cast<int>(count);
    }
    os << '\n';
}
