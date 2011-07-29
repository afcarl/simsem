(function($, window, undefined) {

  var RED_RANGE = 220.0;
  var GREEN_RANGE = 200.0;
  var colour = function(value, range) {
    var r, g, b;
    var mid = (range[0] + range[1]) / 2;
    if (value < mid) {
      b = g = Math.round((value - range[0]) * RED_RANGE / (mid - range[0])) + (255 - RED_RANGE);
      r = 255;
    } else {
      b = r = Math.round((range[1] - value) * GREEN_RANGE / (range[1] - mid)) + (255 - GREEN_RANGE);
      g = 255;
    }
    var rgb = '00000' + $.sprintf('%x', 65536 * r + 256 * g + b);
    return '#' + rgb.substr(rgb.length - 6, 6);
  };

  var addRows = function(rows, formats, ranges, where) {
    $.each(rows, function(rowNo, row) {
      var tr = $('<tr>');
      $.each(row, function(colNo, value) {
        var formattedValue = value === null ? '' : $.sprintf(formats[colNo], value);
        var td = $('<td>' + formattedValue + '</td>');
        if (typeof(value) === 'number') {
          td.css('text-align', 'right');
        }
        var range = ranges[colNo];
        if (range) {
          var rgb = colour(value, range);
          td.css('background-color', rgb);
        }
        tr.append(td);
      });
      where.append(tr);
    });
  };

  var showTable = function(data, side, captionPath) {
    var colNames = data[0];
    var colFormats = data[1];
    var colRanges = data[2];
    var rows = data[3];
    var footerRows = data[4];
    var thead = $('<thead/>');
    var theadTr = $('<tr/>');
    $.each(colNames, function(nameNo, name) {
      theadTr.append('<th>' + name + '</th>');
    });
    thead.append(theadTr);
    var tbody = $('<tbody/>');
    addRows(rows, colFormats, colRanges, tbody);
    var tfoot = $('<tfoot/>');
    addRows(footerRows, colFormats, colRanges, tfoot);
    var table = $('<table class="tab"/>').
        append(thead).
        append(tbody).
        append(tfoot);
    table.tablesorter();
    var desc = $('<div>' + captionPath.join('/') + '</div>');
    var td = $('<td/>').
      append(desc).
      append(table);
    if (side === 0) {
      var tr = $('<tr/>');
      container.append(tr);
    } else {
      var tr = container.children('tr:has(td:only-child):first');
      if (!tr.length) {
        tr = $('<tr><td/></tr>');
        container.append(tr);
      }
    }
    tr.append(td);
  };

  var setPath = function(path, side) {
    var cur = data;
    var startPath = [];
    var restPath = [];
    var varDataTop = null;
    var okay = false;
    for (var i = 0; i < depth; i++) {
      var key = path[i];
      if (!cur) {
        key = '';
      }
      // special processing for multitable requests
      if (key !== undefined) {
        if (key[key.length - 1] === '*') {
          key = key.substr(0, key.length - 1);
          var varDataTop = cur;
          buts[side][i].addClass('starred');
        } else if (varDataTop) {
          restPath.push(key);
        } else {
          startPath.push(key);
        }
      }
      var options = ['<option value="">--Select--</option>'];
      if (cur) {
        $.each(cur, function(key, value) {
          options.push('<option>' + key + '</option>');
        });
        if (!okay) {
          cur = cur[key];
          if (cur instanceof Array) {
            sels[side][i].empty();
            okay = true;
          }
        }
      }
      sels[side][i].html(options.join(''));
      sels[side][i].val(key);
    }
    if (okay) {
      if (varDataTop) {
        $.each(varDataTop, function(curKey, cur) {
          $.each(restPath, function(keyNo, key) {
            cur = cur[key];
          });
          var caption =
              startPath.concat('<span>' + curKey + '</span>', restPath);
          showTable(cur, side, caption);
        });
      } else {
        showTable(cur, side, path);
      }
    }
  };

  var paths = [[], []];
  var updateState = function(evt) {
    container.empty();
    $('#selector input').removeClass('starred');
    var args = $.param.fragment().split(':');
    for (var side = 0; side < 2; side++) {
      paths[side] = args[side] ? args[side].split('/') : [];
      setPath(paths[side], side);
    }
  };

  var findDepth = function(data) {
    if (data instanceof Array) {
      return 0;
    } else {
      var max = 0;
      $.each(data, function(subNo, sub) {
        var subDepth = findDepth(sub);
        if (subDepth > max) {
          max = subDepth;
        }
      });
      return max + 1;
    }
  };

  var updateURL = function() {
    var path = '#' + paths[0].join('/');
    if (paths[1].length) {
      path += ':' + paths[1].join('/');
    }
    location.href = path;
  };

  var selectionChanged = function(evt) {
    var sel = $(evt.target);
    var side = sel.data('side');
    var level = sel.data('level');
    paths[side][level] = sel.val();
    updateURL();
  };

  var allButtonClicked = function(evt) {
    var btn = $(evt.target);
    var side = btn.data('side');
    var level = btn.data('level');
    for (var i = 0; i < depth; i++) {
      var path = paths[side][i];
      if (path[path.length - 1] === '*') {
        paths[side][i] = path.substr(0, path.length - 1);
        if (i === level) {
          // turn off
          updateURL();
          return;
        }
      }
    }
    paths[side][level] += '*';
    updateURL();
  };

  var sels = [[], []];
  buts = [[], []];
  var makeSelection = function(table, side, level) {
    var trs = table.find('tr');

    var selects = $(trs[1]);
    var sel = $('<select data-side="' + side + '" data-level="' + level + '"><option value="">--Select--</option></select>');
    sel.change(selectionChanged);
    sels[side][level] = sel;
    var selTd = $('<td/>').append(sel);
    selects.append(selTd);

    var buttons = $(trs[0]);
    var but = $('<input type="button" value="All" data-side="' + side + '" data-level="' + level + '"/>');
    but.click(allButtonClicked);
    buts[side][level] = but;
    var butTd = $('<td/>').append(but);
    buttons.append(butTd);
  };

  var depth;
  var init = function() {
    container = $('#container');
    depth = findDepth(data);
    var leftContainer = $('#left');
    var rightContainer = $('#right');

    for (var i = 0; i < depth; i++) {
      makeSelection(leftContainer, 0, i);
      makeSelection(rightContainer, 1, i);
    }

    var copyBut = $('<input type="button" value="&#10145"/>');
    var copyTd = $('<td/>').append(copyBut);
    leftContainer.find('tr:eq(1)').append(copyTd);
    copyBut.click(function(evt) {
      paths[1] = paths[0].slice(0); // clone idiom
      updateURL();
    });

    $(window).bind('hashchange', updateState); 
    updateState();
  };

  $(init);
})(jQuery, window);
