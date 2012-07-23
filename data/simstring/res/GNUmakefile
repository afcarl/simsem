# Generate all SimString databases from our lexical resources.
#
# You might want to try `-j ${PROCESSES}` or get a coffee for this one.
#
# Author:	Pontus Stentorp	<pontus stenetorp se>
# Version:	2011-07-25

# XXX: Fragile path
SIMSTRING_BIN:=../../../external/simstring-1.0/frontend/simstring
DB_DIR:=../dbs
DATA_DIR:=simsem_lexical_resources

RES_ARCH=simsem_lexical_resources_2011-07-27T1054Z.tar.gz
RES_URL_ONE=http://www.eidos.ic.i.u-tokyo.ac.jp/~pontus/share/simsem/
RES_URL_TWO=http://udon.stacken.kth.se/~ninjin/share/simsem/
RES_URL_THREE=http://weaver.nlplab.org/~ninjin/share/simsem/

DB_SUFF:=db
# Additional "support" databases created by SimString
CDB_SUFF:=cdb

# Instantiate target variable
DB_TARGETS:=

# Convert path to db-path, create a target for the db and register it globally
define resource-target
RES_PATH:=$1
DB_BASE:=$${shell echo $${RES_PATH} | \
	sed -e 's|${DATA_DIR}/||g' -e 's|^[./]\+||g' -e 's|/|_|g' -e 's|\.|_|g'}
DB_FILE:=$${addsuffix .${DB_SUFF}, $${DB_BASE}}
DB_PATH:=$${addprefix ${DB_DIR}/, $${DB_FILE}}

$${DB_PATH}: $${RES_PATH}
	cat $$< | sort | uniq \
		| ${SIMSTRING_BIN} --build --unicode --database $$@ --quiet

DB_TARGETS+=$${DB_PATH}
DB_TARGETS:=$${sort $${DB_TARGETS}}
endef

# Call the previously declared function for all tokenised resources
${foreach TOKENISED_RESOURCE, \
	${shell find ${DATA_DIR} -name '*.tokens'}, \
	${eval ${call resource-target, ${TOKENISED_RESOURCE}}} \
}

ext_res: ${RES_ARCH}

${RES_ARCH}:
	wget ${RES_URL_ONE}/${RES_ARCH} || wget ${RES_URL_TWO}/${RES_ARCH} || wget ${RES_URL_THREE}/${RES_ARCH}
	tar -x -z -m -f ${RES_ARCH}

all: ${DB_TARGETS}

clean:
	find ${DB_DIR} -maxdepth 1 -name '*.${DB_SUFF}.*.${CDB_SUFF}' \
		-o -name '*.${DB_SUFF}' | xargs -r rm

.PHONY: all clean ext_res
.DEFAULT_GOAL:=all
