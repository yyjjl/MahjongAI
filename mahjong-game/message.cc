#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

#include "message.h"

using Rank = mahjong::rank_t;

Tile ReadTile(std::istream &in) {
    std::string buffer;
    in >> buffer;
    assert(buffer.size() == 2);
    Rank rank = buffer[1] - '0';
    switch (buffer[0]) {
        case 'W': // 万
            return mahjong::make_tile(TILE_SUIT_CHARACTERS, rank);

        case 'B': // 饼
            return mahjong::make_tile(TILE_SUIT_DOTS, rank);

        case 'T': // 条
            return mahjong::make_tile(TILE_SUIT_BAMBOO, rank);

        case 'F': // 风
            return mahjong::make_tile(TILE_SUIT_HONORS, rank);

        case 'J': // 箭
            return mahjong::make_tile(TILE_SUIT_HONORS, 4 + rank);

        default:
            assert(buffer[0] == 'H');
    }
    return kEmptyTile;
}

void WriteTile(std::ostream &os, Tile tile) {
    char type = '?', number = '0' + mahjong::tile_get_rank(tile);
    switch (mahjong::tile_get_suit(tile)) {
        case TILE_SUIT_CHARACTERS:
            type = 'W';
            break;

        case TILE_SUIT_DOTS:
            type = 'B';
            break;

        case TILE_SUIT_BAMBOO:
            type = 'T';
            break;

        case TILE_SUIT_HONORS:
            if (number > '4') {
                type = 'J';
                number -= 4;
            } else
                type = 'F';
            break;

        default:
            assert(tile == kUnknownTile);
            type = 'U';
    }
    os << type << number;
}

void Request::Read(std::istream &in) {
    int message_id;
    in >> message_id;
    switch (message_id) {
        case 0: {
            int t1, t2;
            in >> t1 >> t2;
            men_feng = t1;  // 门风
            quan_feng = t2; // 圈风
            type = kInitSeat;
            break;
        }

        case 1:
            type = kInitTiles;
            break;

        case 2:
            info_tile = ReadTile(in);
            type = kSelfDraw; // 摸到牌
            break;

        default: {
            assert(message_id == 3);

            std::string buffer;
            int tmp;
            in >> tmp >> buffer;
            player_id = tmp;

            assert(buffer.size() >= 3);
            switch (buffer[0]) {
                case 'B': // 补花或补杠
                    info_tile = ReadTile(in);
                    type = buffer[2] == 'G' ? kExtraKong : kExtraFlower;
                    break;

                case 'P': // 出牌或碰牌
                    if (buffer[1] == 'L') {
                        type = kPlay;
                        output_tile = ReadTile(in);
                    } else {
                        type = kPung;
                        output_tile = ReadTile(in);
                    }
                    break;

                case 'C':
                    info_tile = ReadTile(in);
                    output_tile = ReadTile(in);
                    type = kChow;
                    break;

                case 'D': // 其他玩家摸牌
                    type = kDraw;
                    break;

                case 'G': // 明杠或暗杠
                    type = kKong;
                    break;

                default:
                    assert(false); // 错误的指令
            }
        }
    }
}

void Response::Read(std::istream &in, Type request_type) {
    std::string buffer;
    in >> buffer;
    switch (request_type) {
        case kSelfDraw: // 摸牌
            switch (buffer[0]) {
                case 'P':
                    output_tile = ReadTile(in);
                    type = kPlay;
                    break;

                case 'G':
                    info_tile = ReadTile(in);
                    type = kKong;
                    break;

                case 'B':
                    type = kExtraKong;
                    break;

                case 'H':
                    type = kWin;
                    break;

                default:
                    assert(false);
            }
            break;
        case kPlay: // 打
        case kPung: // 碰
        case kChow: // 吃
            switch (buffer[0]) {
                case 'P':
                    if (buffer[1] == 'A')
                        type = kPass;
                    else {
                        output_tile = ReadTile(in);
                        type = kPung;
                    }
                    break;

                case 'C':
                    info_tile = ReadTile(in);
                    output_tile = ReadTile(in);
                    type = kChow;
                    break;

                case 'G':
                    type = kKong;
                    break;

                case 'H':
                    type = kWin;
                    break;

                default:
                    assert(false);
            }
            break;
        default:
            type = kPass;
    }
}

void Response::Write(std::ostream &os) const {
    switch (type) {
        case kPass:
            os << "PASS";
            break;
        case kPlay:
            assert(!IsEmptyTile(output_tile));
            WriteTile(os << "PLAY ", output_tile);
            break;
        case kKong:
            os << "GANG";
            if (!IsEmptyTile(info_tile))
                WriteTile(os << " ", info_tile);
            break;
        case kExtraKong:
            os << "BUGANG";
            break;
        case kPung:
            assert(!IsEmptyTile(output_tile));
            WriteTile(os << "PENG ", output_tile);
            break;
        case kChow:
            assert(!IsEmptyTile(output_tile) && !IsEmptyTile(info_tile));
            WriteTile(os << "CHI ", info_tile);
            WriteTile(os << " ", output_tile);
            break;
        case kWin:
            os << "HU";
            break;
        default:
            os << "???";
    }
}
